/**
 * Simple example network which learns to multiply input by two
 *
 * Input: 5, 3, 8
 * Output: 10, 6, 16
 */

import _ from 'lodash';

import {
  DeferredInputFeed,
  DeferredMemoryInputFeed,
  Dense,
  Model,
  NDArray,
} from '../src';

import { SampleGenerator } from './sample-generator';


export class DenseSimple2x extends SampleGenerator {
  public model(): Model {
    const model = new Model({ seed: this.params.seed });

    model
      .input(1)
      .push(new Dense({ units: 3, activation: 'identity', loss: 'mean-squared-error' }, 'hidden-1'))
      .push(new Dense({ units: 1, activation: 'identity', loss: 'mean-squared-error' }, 'output'));

    return model;
  }


  public samples(count: number): DeferredInputFeed {
    return DeferredMemoryInputFeed.factory(
      _.times(count, n => ({ x: new NDArray([(n + 1)]), y: new NDArray([(n + 1) * 2]) })),
    );
  }
}
