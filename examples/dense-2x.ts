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

    model
      .input(1)
      .push(new Dense({ units: 4, activation: 'relu' }))
      .push(new Dense({ units: 4, activation: 'relu' }))
      .push(new Dense({ units: 1, activation: 'sigmoid' }, 'result'));

    // .output('result'); /* should be automatic, unless you want to declare it */

    return model;
  }


  public samples(count: number): DeferredInputFeed {
    return DeferredMemoryInputFeed.factory(
      _.times(count, n => ({ x: new NDArray([n]), y: new NDArray([n * 2]) })),
    );
  }
}
