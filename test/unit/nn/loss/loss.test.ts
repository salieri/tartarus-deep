// tslint:disable no-duplicate-imports
/* eslint-disable import/no-duplicates */

import _ from 'lodash';

import * as losses from '../../../../src/nn/loss';
import { Loss } from '../../../../src/nn/loss';
import { Vector } from '../../../../src/math';

describe(
  'Loss',
  () => {
    it(
      'should instantiate loss classes',
      () => {
        let instanceCount = 0;

        _.each(
          losses,
          (L) => {
            if (!(L.prototype instanceof Loss)) {
              return;
            }

            const instance = new L();
            const y = new Vector([-5.102, -3.2129, 291, 0, 29.1, 1.123, 3.21, -0.291, 12.2, 17.2]);
            const yHat = new Vector([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]);

            instance.calculate(yHat, y).should.be.a('number');
            instance.gradient(yHat, y).should.be.an.instanceOf(Vector);

            instanceCount += 1;
          },
        );

        instanceCount.should.be.greaterThan(10);
      },
    );
  },
);

