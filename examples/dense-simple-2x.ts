/**
 * Simple example network which learns to multiply input by two
 *
 * Input: 5, 3, 8
 * Output: 10, 6, 16
 */

import _ from 'lodash';

import {
  MemoryInputFeed,
  Dense,
  Model,
  NDArray,
} from '../src';

import { SampleGenerator } from './sample-generator';
import { Stochastic } from '../src/nn/optimizer';


export class DenseSimple2x extends SampleGenerator {
  public model(): Model {
    const optimizer = new Stochastic({ rate: 0.002 });
    const model = new Model({ optimizer, seed: this.params.seed, loss: 'mean-squared-error' });

    model
      .input(1)
      .push(new Dense({ units: 3, activation: 'identity', bias: false }, 'hidden-1'))
      .push(new Dense({ units: 1, activation: 'identity', bias: false }, 'output'));

    return model;
  }


  public samples(count: number): MemoryInputFeed {
    return MemoryInputFeed.factory(
      _.times(
        count,
        (n: number) => {
          const x = n % 25;

          return { x: new NDArray([x]), y: new NDArray([x * 2]) };
        },
      ),
    );
  }
}
