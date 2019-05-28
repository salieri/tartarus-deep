// tslint:disable no-duplicate-imports
/* eslint-disable import/no-duplicates */

import _ from 'lodash';
import Promise from 'bluebird';

import * as initializers from '../../../../src/nn/initializer';
import { Initializer } from '../../../../src/nn/initializer';
import { NDArray } from '../../../../src/math';
import { Dense, Session } from '../../../../src/nn';

describe(
  'Initializer',
  () => {
    it(
      'should instantiate activation classes',
      async () => {
        let instanceCount = 0;

        await Promise.props(
          _.mapValues(
            initializers as any,
            async (I: any) => {
              if (!(I.prototype instanceof Initializer)) {
                return;
              }

              instanceCount += 1;

              const instance = new I();
              const data = new NDArray([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]);
              const dense = new Dense({ units: 1 });
              const session = new Session();

              dense.setSession(session);
              instance.attachLayer(dense);

              (await instance.initialize(data)).should.be.instanceOf(NDArray);
            },
          ),
        );

        instanceCount.should.be.greaterThan(3);
      },
    );
  },
);

