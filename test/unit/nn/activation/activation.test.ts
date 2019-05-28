// tslint:disable no-duplicate-imports
/* eslint-disable import/no-duplicates */

import _ from 'lodash';

import * as activations from '../../../../src/nn/activation';
import { Activation } from '../../../../src/nn/activation';
import { NDArray } from '../../../../src/math';

describe(
  'Activation',
  () => {
    it(
      'should instantiate activation classes',
      () => {
        let instanceCount = 0;

        _.each(
          activations,
          (A) => {
            if (!(A.prototype instanceof Activation)) {
              return;
            }

            const instance = new A();
            const data = new NDArray([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]);

            instance.calculate(data).should.be.instanceOf(NDArray);

            instanceCount += 1;
          },
        );

        instanceCount.should.be.greaterThan(15);
      },
    );
  },
);

