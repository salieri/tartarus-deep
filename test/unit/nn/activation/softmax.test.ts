import * as activations from '../../../../src/nn/activation';
import { NDArray } from '../../../../src/math';


describe.only(
  'Activation: Softmax',
  () => {
    it(
      'should calculate Softmax activation',
      () => {
        const sa = new activations.Softmax();
        const input = new NDArray([1, 2, 3]);
        const output = sa.calculate(input);

        output.getAt([0]).should.be.closeTo(0.09, 0.01);
        output.getAt([1]).should.be.closeTo(0.24, 0.01);
        output.getAt([2]).should.be.closeTo(0.66, 0.01);

        const input2 = new NDArray([1, 2]);
        const output2 = sa.calculate(input2);

        output2.getAt([0]).should.be.closeTo(0.26, 0.01);
        output2.getAt([1]).should.be.closeTo(0.73, 0.01);
      },
    );


    it(
      'should deal with big numbers in Softmax activation',
      () => {
        const sa = new activations.Softmax();
        const input = new NDArray([1000, 2000, 3000]);
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
        const input = new NDArray([1, 2, 3]);
        const output = sa.calculate(input);

      },
    );
  },
);
