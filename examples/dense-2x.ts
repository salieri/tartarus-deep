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
      .input(1)
      .push(new Dense({ units: 4, activation: 'relu' }, 'dense-1'))
      .push(new Dense({ units: 4, activation: 'relu' }, 'dense-2'))
      .push(new Dense({ units: 1, activation: 'sigmoid' }, 'result'))
      .output('result');


    model
      .add(new Concat('moo'), ['dense-1', 'dense-2'])
      .add(new Dense(), ['moo']);

    return model;
  }

  public samples(count: number): SampleData[] {
    return _.times(count, n => ({ x: new NDArray([n]), y: new NDArray([n * 2]) }));
  }
}
