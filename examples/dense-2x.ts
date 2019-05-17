/**
 * Simple example network which learns to multiply input by two
 *
 * Input: 5, 3, 8
 * Output: 10, 6, 16
 */

import _ from 'lodash';

import {
  Dense,
  Model,
  NDArray,
} from '../src';

import { SampleData, SampleGenerator } from './sample-generator';


export class Dense2x extends SampleGenerator {
  public model(): Model {
    const model = new Model({ seed: this.params.seed });

    model
      .add(new Dense({ units: 4, activation: 'relu' }))
      .add(new Dense({ units: 4, activation: 'relu' }))
      .add(new Dense({ units: 1, activation: 'sigmoid' }));

    return model;
  }

  public samples(count: number): SampleData[] {
    return _.times(count, n => ({ x: new NDArray([n]), y: new NDArray([n * 2]) }));
  }
}
