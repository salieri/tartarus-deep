import _ from 'lodash';

import { ContextLogger, JoiEx, JoiExSchema } from '../../util';
import { Parameters, Parameterized } from '../../generic';
import { Model } from './model';
import { DeferredInputFeed } from '../../feed';
import { DeferredInputCollection } from '../symbols/deferred';
import { NDArray } from '../../math';
import { GraphNode } from '../graph';


export interface FitterParams extends Parameters {
  batchSize?: number;
  epochs?: number;
}


export interface BatchResult {
  iterations: number;
  avgLoss: number;
  dError: DeferredInputCollection;
}


/**
 * @link https://www.youtube.com/watch?v=G5b4jRBKNxw
 * @link https://www.youtube.com/watch?v=Zr5viAZGndE
 * @link https://www.youtube.com/watch?v=xClK__CqZnQ
 * @link https://stats.stackexchange.com/questions/154879/a-list-of-cost-functions-used-in-neural-networks-alongside-applications
 * @link https://towardsdatascience.com/step-by-step-tutorial-on-linear-regression-with-stochastic-gradient-descent-1d35b088a843
 */

export class ModelFitter extends Parameterized<FitterParams> {
  protected model: Model;

  protected feed: DeferredInputFeed;

  protected logger: ContextLogger;


  public constructor(model: Model, feed: DeferredInputFeed, params: FitterParams) {
    super(params);

    if (!feed.hasLabels()) {
      throw new Error('Cannot fit data that has no defined labels');
    }

    this.model = model;
    this.feed = feed;

    this.logger = new ContextLogger(model.getSession().getLogger(), 'connector');
  }


  protected async fitBatch(): Promise<BatchResult> {
    let result: DeferredInputCollection|undefined;
    let iterations: number;
    let totalLoss = 0;

    for (iterations = 0; (iterations < this.params.batchSize) && (this.feed.hasMore()); iterations += 1) {
      /* eslint-disable-next-line no-await-in-loop */
      const data = await this.feed.next();

      if (!data.label) {
        throw new Error('Cannot fit data if labels are not defined');
      }

      /* eslint-disable-next-line no-await-in-loop */
      const iterationResult = await this.model.iterate(data.sample.raw, data.label.raw);

      // console.log(`${data.sample.raw.getDefaultValue().sum()}  * 2 = ${iterationResult.prediction.getDefaultValue().sum()}`);

      const iterationFit = new DeferredInputCollection();

      _.each(
        this.model.getGraph().getAllNodes(),
        (node: GraphNode) => {
          iterationFit.set(node.getName(), node.getEntity().data.fitter.clone());
        },
      );

      // dWTotal += dW[iteration]
      // dbTotal += db[iteration]
      result = this.sumResults(result, iterationFit);
      totalLoss += iterationResult.loss.getDefaultValue().sum();

      this.logger.debug(
        'fit.epoch.batch.iteration',
        () => ({
          curIteration: (iterations + 1),
          batchSize: this.params.batchSize,
          loss: iterationResult.loss,
          input: data.sample.raw,
          output: iterationResult.prediction,
        }),
      );
    }

    if (!result) {
      throw new Error('Did not iterate over any data');
    }

    // dWTotal /= m, dbTotal /= m
    result.eachValue(<T extends NDArray> (val: T): T => val.div(iterations));

    // reassign dW, db
    this.reassignFitValues(result);

    await this.model.optimize();

    return {
      iterations,
      avgLoss: totalLoss / iterations,
      dError: result,
    };
  }


  protected sumResults(totalResults: DeferredInputCollection|undefined, iterationResult: DeferredInputCollection): DeferredInputCollection {
    if (!totalResults) {
      return iterationResult.clone();
    }

    totalResults.eachValue(
      /* eslint-disable-next-line arrow-parens */
      <T extends NDArray> (curVal: T, collectionKey: string, fieldKey: string): T => {
        const iterationVal = iterationResult.get(collectionKey).getValue(fieldKey);

        return curVal.add(iterationVal);
      },
    );

    return totalResults;
  }


  protected reassignFitValues(result: DeferredInputCollection): void {
    result.eachValue(
      /* eslint-disable-next-line arrow-parens */
      <T extends NDArray> (val: T, collectionKey: string, fieldKey: string): void => {
        const node = this.model.getGraph().find(collectionKey);

        node.getEntity().data.fitter.setValue(fieldKey, val.clone());
      },
    );
  }


  protected async fitEpoch(curEpoch: number): Promise<void> {
    let batchCount = 0;
    let prevLoss = 10000000;

    await this.feed.seek(0);

    while (this.feed.hasMore()) {
      /* eslint-disable-next-line no-await-in-loop */
      const result = await this.fitBatch();

      // console.log('AVGLoss', curEpoch, batchCount, result.avgLoss, Math.abs(result.avgLoss) < Math.abs(prevLoss) ? 'Better' : 'Worse');

      this.logger.info(
        'fit.epoch.batch',
        () => ({
          curEpoch: (curEpoch + 1),
          curBatch: (batchCount + 1),
          avgLoss: result.avgLoss,
          iterations: result.iterations,
          dError: result.dError,
        }),
      );

      batchCount += 1;

      prevLoss = result.avgLoss;
    }
  }


  public async fit(): Promise<void> {
    for (let curEpoch = 0; curEpoch < this.params.epochs; curEpoch += 1) {
      /* eslint-disable-next-line no-await-in-loop */
      await this.fitEpoch(curEpoch);

      this.logger.info('fit.epoch', () => ({ curEpoch: (curEpoch + 1), epochTotal: this.params.epochs }));
    }
  }


  public getParamSchema(): JoiExSchema {
    return JoiEx.object().keys(
      {
        batchSize: JoiEx.number().optional().default(32).description('Number of samples per gradient update'),
        epochs: JoiEx.number().optional().default(1).description('Number of epochs to train the model'),
      },
    );
  }
}
