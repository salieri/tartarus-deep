// tslint:disable no-duplicate-imports
/* eslint-disable import/no-duplicates */

import _ from 'lodash';

import * as metrics from '../../../../src/nn/metric';
import { Metric } from '../../../../src/nn/metric';
import { Vector } from '../../../../src/math';

describe(
  'Metric',
  () => {
    it(
      'should instantiate metric classes',
      () => {
        let instanceCount = 0;

        _.each(
          metrics,
          (M) => {
            if (!(M.prototype instanceof Metric)) {
              return;
            }

            const instance = new M();
            const y = new Vector([-5.102, -3.2129, 291, 0, 29.1, 1.123, 3.21, -0.291, 12.2, 17.2]);
            const yHat = new Vector([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]);

            instance.calculate(yHat, y).should.be.a('number');

            instanceCount += 1;
          },
        );

        instanceCount.should.be.greaterThan(3);
      },
    );
  },
);

