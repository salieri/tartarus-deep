import _ from 'lodash';

import { ContextLogger, JoiEx, JoiExSchema } from '../../util';
import { Parameters, Parameterized } from '../../generic';
import { Model } from './model';
import { DeferredInputFeed } from '../../feed';
import { DeferredCollection, DeferredInputCollection } from '../symbols/deferred';
import { NDArray, Vector, VectorDirection } from '../../math';
import { GraphNode } from '../graph';
import { MeanSquaredError } from '../loss';
import { Dense, Layer } from '../layer';

export interface FitterParams extends Parameters {
  batchSize?: number;
  epochs?: number;
}


/**
 * @link https://www.youtube.com/watch?v=G5b4jRBKNxw
 * @link https://www.youtube.com/watch?v=Zr5viAZGndE
 * @link https://www.youtube.com/watch?v=xClK__CqZnQ
 * @link https://stats.stackexchange.com/questions/154879/a-list-of-cost-functions-used-in-neural-networks-alongside-applications
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


  protected async fitBatch(): Promise<void> {
    let result: DeferredInputCollection|undefined;
    let iterations: number;

    for (iterations = 0; (iterations < this.params.batchSize) && (this.feed.hasMore()); iterations += 1) {
      /* eslint-disable-next-line no-await-in-loop */
      const data = await this.feed.next();

      if (!data.label) {
        throw new Error('Cannot fit data if labels are not defined');
      }

      /* eslint-disable-next-line no-await-in-loop */
      await this.model.iterate(data.sample.raw, data.label.raw);

      const iterationFit = new DeferredInputCollection();

      _.each(
        this.model.getGraph().getAllNodes(),
        (node: GraphNode) => {
          // iterationFit.set(node.getName(), node.getEntity().data.fitter.clone());

          const dErrorOverLinear = node.getEntity().data.backpropOutput.getValue(Layer.ERROR_TERM, Vector);
          const output = node.getEntity().data.output.getDefaultValue(Vector);

          const tw = dErrorOverLinear.mul(output);
          const tb = new Vector([dErrorOverLinear.sum()]);

          const actualWE = node.getEntity().data.fitter.getValue(Dense.WEIGHT_ERROR);
          const actualBE = node.getEntity().data.fitter.getValue(Dense.BIAS_ERROR);

          const coll = new DeferredCollection();

          coll.declare(Dense.WEIGHT_ERROR, actualWE.getDims());
          coll.declare(Dense.BIAS_ERROR, actualBE.getDims());

          coll.setValue(Dense.WEIGHT_ERROR, tw.expandToMatrix(1, 3, VectorDirection.Col).transpose());
          coll.setValue(Dense.BIAS_ERROR, tb);

          iterationFit.set(node.getName(), coll);
        },
      );

      // dWTotal += dW[iteration]
      // dbTotal += db[iteration]
      result = this.sumResults(result, iterationFit);

console.log('------------------- Iteration --------------------');
console.log(`input: ${data.sample.raw.getDefaultValue().getAt(0)}`);
console.log(`output: ${this.model.getGraph().find('output').getEntity().data.output.getDefaultValue().getAt(0)}`);
console.log(`SUM weight-error: ${this.model.getGraph().find('output').getEntity().data.fitter.getValue('weight-error').sum()}`);
console.log(`weight-error: ${this.model.getGraph().find('output').getEntity().data.fitter.getValue('weight-error').data}`);
console.log(`SUM error-term: ${this.model.getGraph().find('output').getEntity().data.backpropOutput.getValue('error-term').sum()}`);
console.log(`error-term: ${this.model.getGraph().find('output').getEntity().data.backpropOutput.getValue('error-term').data}`);
console.log('');

      this.logger.debug(
        'fit.epoch.batch.iteration',
        () => ({ curIteration: (iterations + 1), totalIterations: this.params.batchSize }),
      );
    }

    if (!result) {
      throw new Error('Did not iterate over any data');
    }

console.log('');
console.log('================= Batch result ===================');

console.log(`SUM result-total: ${result.get('output').getValue('weight-error').sum()}`);


const loss = new MeanSquaredError();

const avgLoss = loss.calculate(
  this.model.getGraph().find('output').getEntity().data.output.getValue('activated'),
  this.model.getGraph().find('output').getEntity().data.trainer.getDefaultValue(),
);

    // dWTotal /= m, dbTotal /= m
    result.eachValue(<T extends NDArray> (val: T): T => val.div(iterations));


console.log(`SUM result-divided: ${result.get('output').getValue('weight-error').sum()}`);

    // reassign dW, db
    this.reassignFitValues(result);

console.log(`SUM weight-error-divided: ${this.model.getGraph().find('output').getEntity().data.fitter.getValue('weight-error').sum()}`);

    await this.model.optimize();
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


  protected async fitEpoch(): Promise<void> {
    let batchCount = 0;

    await this.feed.seek(0);

    while (this.feed.hasMore()) {
      /* eslint-disable-next-line no-await-in-loop */
      await this.fitBatch();

      this.logger.info(
        'fit.epoch.batch',
        () => ({ curBatch: (batchCount + 1) }),
      );

      batchCount += 1;
    }
  }


  public async fit(): Promise<void> {
    for (let curEpoch = 0; curEpoch < this.params.epochs; curEpoch += 1) {
      /* eslint-disable-next-line no-await-in-loop */
      await this.fitEpoch();

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
