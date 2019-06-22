/* eslint-disable no-loop-func, no-await-in-loop */

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


export interface FitResult {
  epochs: number;
  batches: number;
  iterations: number;
  loss: LossResult;
}


export interface LossResult {
  average: number;
  final: number;
  first: number;
  min: number;
  max: number;
}


export interface BatchResult {
  iterations: number;
  curEpoch: number;
  curBatch: number;
  loss: LossResult;
}


export interface EpochResult {
  curEpoch: number;
  batches: number;
  iterations: number;
  loss: LossResult;
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


  protected async fitBatch(curEpoch: number, curBatch: number): Promise<BatchResult> {
    let result: DeferredInputCollection|undefined;
    let iterations: number;
    let totalLoss = 0;

    let finalLoss = Number.MAX_VALUE;
    let minLoss = Number.MAX_VALUE;
    let maxLoss = Number.MIN_VALUE;
    let firstLoss: number|undefined;

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

      finalLoss = iterationResult.loss.getDefaultValue().sum();
      minLoss = Math.min(minLoss, finalLoss);
      maxLoss = Math.max(maxLoss, finalLoss);

      totalLoss += finalLoss;

      if (firstLoss === undefined) {
        firstLoss = finalLoss;
      }

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

    if ((!result) || (iterations === 0)) {
      throw new Error('Did not iterate over any data');
    }

    // dWTotal /= m, dbTotal /= m
    result.eachValue(<T extends NDArray> (val: T): T => val.div(iterations));

    // reassign dW, db
    this.reassignFitValues(result);

    await this.model.optimize();

    const batchResult: BatchResult = {
      curEpoch,
      curBatch,
      iterations,
      loss: {
        first: firstLoss as number,
        average: totalLoss / iterations,
        min: minLoss,
        max: maxLoss,
        final: finalLoss,
      },
    };

    this.logger.info('fit.epoch.batch', batchResult);

    return batchResult;
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


  protected async fitEpoch(curEpoch: number): Promise<EpochResult> {
    let curBatch = 0;
    let lastResult: BatchResult|undefined;

    let totalAverageLoss = 0;
    let minLoss = Number.MAX_VALUE;
    let maxLoss = Number.MIN_VALUE;
    let firstLoss: number|undefined;
    let totalIterations = 0;

    await this.feed.seek(0);

    while (this.feed.hasMore()) {
      /* eslint-disable-next-line no-await-in-loop */
      lastResult = await this.fitBatch(curEpoch, curBatch);

      totalAverageLoss += lastResult.loss.average;
      totalIterations += lastResult.iterations;

      if (firstLoss === undefined) {
        firstLoss = lastResult.loss.average;
      }

      minLoss = Math.min(minLoss, lastResult.loss.average);
      maxLoss = Math.max(maxLoss, lastResult.loss.average);

      // console.log('AVGLoss', curEpoch, batchCount, result.avgLoss, Math.abs(result.avgLoss) < Math.abs(prevLoss) ? 'Better' : 'Worse');
      curBatch += 1;
    }

    if ((curBatch === 0) || (!lastResult)) {
      throw new Error('Did not fit any batches');
    }

    const epochResult: EpochResult = {
      curEpoch,
      batches: curBatch,
      iterations: totalIterations,
      loss: {
        first: firstLoss as number,
        final: lastResult.loss.final as number,
        average: totalAverageLoss / curBatch,
        min: minLoss,
        max: maxLoss,
      },
    };

    this.logger.info('fit.epoch', epochResult);

    return epochResult;
  }

  public async fit(): Promise<FitResult> {
    let totalBatches = 0;
    let totalIterations = 0;
    let totalAverageLoss = 0;

    let minLoss = Number.MAX_VALUE;
    let maxLoss = Number.MIN_VALUE;

    let epochResult: EpochResult|undefined;
    let firstLoss: number|undefined;

    for (let curEpoch = 0; curEpoch < this.params.epochs; curEpoch += 1) {
      /* eslint-disable-next-line no-await-in-loop */
      epochResult = await this.fitEpoch(curEpoch);

      totalBatches += epochResult.batches;
      totalIterations += epochResult.iterations;
      totalAverageLoss += epochResult.loss.average;

      minLoss = Math.min(minLoss, epochResult.loss.min);
      maxLoss = Math.max(maxLoss, epochResult.loss.max);

      if (firstLoss === undefined) {
        firstLoss = epochResult.loss.first;
      }
    }


    if (!epochResult) {
      throw new Error('Did not fit any epochs');
    }

    const fitResult: FitResult = {
      epochs: this.params.epochs,
      batches: totalBatches,
      iterations: totalIterations,
      loss: {
        first: firstLoss as number,
        final: epochResult.loss.final,
        average: totalAverageLoss / this.params.epochs,
        min: minLoss,
        max: maxLoss,
      },
    };

    this.logger.info('fit', fitResult);

    return fitResult;
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
