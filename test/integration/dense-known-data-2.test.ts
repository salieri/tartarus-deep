import { Dense, Layer, Model } from '../../src/nn';
import { Matrix, Vector } from '../../src/math';

/**
 * @link https://www.anotsorandomwalk.com/backpropagation-example-with-numbers-step-by-step/
 */
describe(
  'Dense Model with Known (Precalculated) Data 2',
  () => {
    const m = new Model();
    const h = new Dense({ units: 2, activation: 'sigmoid' });
    const o = new Dense({ units: 2, activation: 'sigmoid' });

    it(
      'should declare and compile a model',
      async () => {
        m.input(3)
          .push(h)
          .push(o);

        await m.compile();
        await m.initialize();
      },
    );


    it(
      'should perform forward pass',
      async () => {
        h.optimizer.setValue(Dense.WEIGHT_MATRIX, new Matrix([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]));
        h.optimizer.setValue(Dense.BIAS_VECTOR, new Vector([0.5]));

        o.optimizer.setValue(Dense.WEIGHT_MATRIX, new Matrix([[0.7, 0.9], [0.8, 0.1]]));
        o.optimizer.setValue(Dense.BIAS_VECTOR, new Vector([0.5]));

        await m.predict([1, 4, 5]);

        const hOut = h.output.getDefaultValue();
        const oOut = o.output.getDefaultValue();

        hOut.getAt([0]).should.be.closeTo(0.9866, 0.0001);
        hOut.getAt([1]).should.be.closeTo(0.9950, 0.0001);

        oOut.getAt([0]).should.be.closeTo(0.8896, 0.0001);
        oOut.getAt([1]).should.be.closeTo(0.8004, 0.0001);
      },
    );


    it(
      'should perform backward pass',
      async () => {
        h.optimizer.setValue(Dense.WEIGHT_MATRIX, new Matrix([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]));
        h.optimizer.setValue(Dense.BIAS_VECTOR, new Vector([0.5]));

        o.optimizer.setValue(Dense.WEIGHT_MATRIX, new Matrix([[0.7, 0.9], [0.8, 0.1]]));
        o.optimizer.setValue(Dense.BIAS_VECTOR, new Vector([0.5]));

        await m.fit([1, 4, 5], [0.1, 0.05]);

        const oBack = o.backpropOutput.getValue(Layer.DERIVATIVE);
        const hBack = h.backpropOutput.getValue(Layer.DERIVATIVE);

        oBack.getDims().should.deep.equal([2]);
        oBack.getAt(0).should.be.closeTo(0.0775735, 0.00001);
        oBack.getAt(1).should.be.closeTo(0.1198838, 0.00001);

        hBack.getDims().should.deep.equal([2]);
        hBack.getAt(0).should.be.closeTo(0.00198264, 0.00001);
        hBack.getAt(1).should.be.closeTo(0.00040082, 0.00001);

        const odWeight = o.calculateWeightDerivative(new Vector(oBack));
        const odBias = o.calculateBiasDerivative(new Vector(oBack));

        odWeight.getDims().should.deep.equal([2, 2]);
        odWeight.getAt([0, 0]).should.be.closeTo(0.0765, 0.0001);
        odWeight.getAt([0, 1]).should.be.closeTo(0.1183, 0.0001);
        odWeight.getAt([1, 0]).should.be.closeTo(0.0772, 0.0001);
        odWeight.getAt([1, 1]).should.be.closeTo(0.1193, 0.0001);
        odBias.should.be.closeTo(0.1975, 0.001);

        const hdWeight = h.calculateWeightDerivative(new Vector(hBack));
        const hdBias = h.calculateBiasDerivative(new Vector(hBack));

        hdWeight.getDims().should.deep.equal([3, 2]);
        hdWeight.getAt([0, 0]).should.be.closeTo(0.002, 0.0001);
        hdWeight.getAt([1, 0]).should.be.closeTo(0.0079, 0.0001);
        hdWeight.getAt([2, 0]).should.be.closeTo(0.0099, 0.0001);
        hdWeight.getAt([0, 1]).should.be.closeTo(0.0004, 0.0001);
        hdWeight.getAt([1, 1]).should.be.closeTo(0.0016, 0.0001);
        hdWeight.getAt([2, 1]).should.be.closeTo(0.0020, 0.0001);
        // hdBias.should.be.closeTo(0.0008, 0.0001);
      },
    );
  },
);
