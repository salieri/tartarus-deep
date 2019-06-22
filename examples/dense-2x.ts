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


export class Dense2x extends SampleGenerator {
  public model(): Model {
    const model = new Model({ seed: this.params.seed });

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


  public samples(count: number): DeferredInputFeed {
    return DeferredMemoryInputFeed.factory(
      _.times(count, n => ({ x: new NDArray([n]), y: new NDArray([n * 2]) })),
    );
  }
}
