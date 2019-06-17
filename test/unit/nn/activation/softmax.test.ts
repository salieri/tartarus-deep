import * as activations from '../../../../src/nn/activation';
import { Vector } from '../../../../src/math';

describe(
  'Activation: Softmax',
  () => {
    it(
      'should calculate Softmax activation',
      () => {
        const sa = new activations.Softmax();
        const input = new Vector([1, 2, 3]);
        const output = sa.calculate(input);

        output.getAt([0]).should.be.closeTo(0.09, 0.01);
        output.getAt([1]).should.be.closeTo(0.24, 0.01);
        output.getAt([2]).should.be.closeTo(0.66, 0.01);

        const input2 = new Vector([1, 2]);
        const output2 = sa.calculate(input2);

        output2.getAt([0]).should.be.closeTo(0.26, 0.01);
        output2.getAt([1]).should.be.closeTo(0.73, 0.01);
      },
    );


    it(
      'should deal with big numbers in Softmax activation',
      () => {
        const sa = new activations.Softmax();
        const input = new Vector([1000, 2000, 3000]);
        const output = sa.calculate(input);

        output.getAt([0]).should.be.closeTo(0, 0.01);
        output.getAt([1]).should.be.closeTo(0, 0.01);
        output.getAt([2]).should.be.closeTo(1, 0.01);
      },
    );


    it(
      'should calculate Softmax derivatives',
      () => {
        const sa = new activations.Softmax();
        const input = new Vector([-1, -1, 1]);
        const softmax = sa.calculate(input);
        const target = new Vector([0, 1, 0]);
        const derivative = sa.derivative(softmax, input, target);
        const softmaxMinusTarget = softmax.sub(target); // should equal derivative

        derivative.iterate(
          (vals: number[]): number => {
            vals[0].should.be.closeTo(vals[1], 0.000001);
            return 0;
          },
          softmaxMinusTarget,
        );
      },
    );
  },
);
