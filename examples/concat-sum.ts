/**
 * Simple example network which demonstrates the use of CONCAT layer and learns to
 * sum the input.
 *
 * E.g.:
 * Input: a = 5, b = [3, 8]
 * Output: 5 + 3 + 8 = 16
 */

import _ from 'lodash';

import {
  MemoryInputFeed,
  Concat,
  Dense,
  Model,
  NDArray, DeferredInputCollection, DeferredCollection, Xoshiro128,
} from '../src';

import { SampleGenerator } from './sample-generator';
import { Stochastic } from '../src/nn/optimizer';


export class ConcatSum extends SampleGenerator {
  public model(): Model {
    const optimizer = new Stochastic({ rate: 0.001 });
    const model = new Model({ optimizer, seed: this.params.seed, loss: 'mean-squared-error' });

    /**
     * An intentionally unnecessarily complex network for figuring out a + b
     */

    const inputDefinition = new DeferredInputCollection();
    const inputA = new DeferredCollection();
    const inputB = new DeferredCollection();

    inputA.declareDefault(1);
    inputB.declareDefault(2);

    inputDefinition.set('a', new DeferredCollection());
    inputDefinition.set('b', new DeferredCollection());

    model
      .input(inputDefinition)
      .add(new Dense({ units: 3, activation: 'identity' }, 'a'))
      .add(new Dense({ units: 4, activation: 'identity' }, 'b'))
      .add(new Concat({}, 'concat'), ['a', 'b'])
      .add(new Dense({ units: 1, activation: 'identity' }, 'output'), 'concat')
      .output('output');

    return model;
  }


  public samples(count: number): MemoryInputFeed {
    const r = new Xoshiro128(this.params.seed);

    return MemoryInputFeed.factory(
      _.times(
        count,
        () => {
          const a = r.floatBetween(1, 20);
          const b1 = r.floatBetween(1, 5);
          const b2 = r.floatBetween(1, 12);

          return {
            x: {
              a: new NDArray([a]),
              b: new NDArray([b1, b2]),
            },
            y: new NDArray([a + b1 + b2]),
          };
        },
      ),
    );
  }
}

