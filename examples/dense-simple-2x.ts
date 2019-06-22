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
  Xoshiro128,
} from '../src';

import { SampleGenerator } from './sample-generator';


export class DenseSimple2x extends SampleGenerator {
  public model(): Model {
    const model = new Model({ seed: this.params.seed, loss: 'mean-squared-error' });

    model
      .input(1)
      .push(new Dense({ units: 3, activation: 'identity', bias: false }, 'hidden-1'))
      .push(new Dense({ units: 1, activation: 'identity', bias: false }, 'output'));

    return model;
  }


  public samples(count: number): DeferredInputFeed {
    return DeferredMemoryInputFeed.factory(
      _.times(
        count,
        (n: number) => {
          const x = n % 10;

          return { x: new NDArray([x]), y: new NDArray([x * 2]) };
        },
      ),
    );
  }
}
