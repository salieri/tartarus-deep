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


export class Dense2x extends SampleGenerator {
  public model(): Model {
    const optimizer = new Stochastic({ rate: 0.001 });
    const model = new Model({ optimizer, seed: this.params.seed, loss: 'mean-squared-error' });

    /**
     * An intentionally unnecessarily complex network for figuring out 2X
     */
    model
      .input(1)
      .push(new Dense({ units: 4, activation: 'identity' }, 'hidden-1'))
      .push(new Dense({ units: 5, activation: 'identity' }, 'hidden-2'))
      .push(new Dense({ units: 1, activation: 'identity' }, 'output'));

    // .output('result'); /* will be automatically set to the last node, unless you choose to declare it */

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
