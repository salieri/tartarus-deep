// tslint:disable no-duplicate-imports
/* eslint-disable import/no-duplicates */

import _ from 'lodash';

import * as activations from '../../../../src/nn/activation';
import { Activation } from '../../../../src/nn/activation';
import { Vector } from '../../../../src/math';

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
            const z = new Vector([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]);
            const y = new Vector([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]);
            const a = instance.calculate(z);
            const da = instance.gradient(a, z, y);

            a.should.be.instanceOf(Vector);
            da.should.be.instanceOf(Vector);

            instanceCount += 1;
          },
        );

        instanceCount.should.be.greaterThan(15);
      },
    );

    it(
      'should produce verifiable gradients',
      () => {
        _.each(
          activations,
          (A, k: string) => {
            if (!(A.prototype instanceof Activation)) {
              return;
            }

            // This is not a suitable test for Softmax
            if (k === 'Softmax') {
              return;
            }

            const instance = new A();

            const e = 0.000001;
            const z = new Vector([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]);
            const y = new Vector([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]);

            const zpe = z.add(e);
            const zme = z.sub(e);

            const a = instance.calculate(z);
            const ape = instance.calculate(zpe);
            const ame = instance.calculate(zme);

            const da = instance.gradient(a, z, y);
            const dApprox = ape.sub(ame).div(e * 2);

            da.iterate(
              (vals: number[]): number => {
                const daVal = vals[0];
                const dApproxVal = vals[1];

                try {
                  dApproxVal.should.be.closeTo(daVal, 0.000001);
                } catch (err) {
                  console.error(`Expected approximated value ${dApproxVal} to be close to ${daVal} for activation '${k}'`);
                  throw err;
                }

                return 0;
              },
              dApprox,
            );
          },
        );
      },
    );
  },
);

