// tslint:disable no-duplicate-imports
/* eslint-disable import/no-duplicates */

import _ from 'lodash';

import * as costs from '../../../../src/nn/cost';
import { Cost } from '../../../../src/nn/cost';
import { NDArray } from '../../../../src/math';

describe(
  'Cost',
  () => {
    it(
      'should instantiate cost classes',
      () => {
        let instanceCount = 0;

        _.each(
          costs,
          (C) => {
            if (!(C.prototype instanceof Cost)) {
              return;
            }

            const instance = new C();
            const data = new NDArray([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]);

            instance.calculate(data).should.be.a('number');

            instanceCount += 1;
          },
        );

        instanceCount.should.be.greaterThan(0);
      },
    );
  },
);

