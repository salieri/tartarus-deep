import { Dense, Layer, Model } from '../../src/nn';
import { Matrix, Vector } from '../../src/math';

describe.only(
  'Dense Model with Known (Precalculated) Data',
  () => {
    const m = new Model();
    const h = new Dense({ units: 2, activation: 'sigmoid' });
    const o = new Dense({ units: 2, activation: 'sigmoid' });

    it(
      'should declare and compile a model',
      async () => {
        m.input(2)
          .push(h)
          .push(o);

        await m.compile();
        await m.initialize();
      },
    );


    it(
      'should perform forward pass',
      async () => {
        h.optimizer.setValue(Dense.WEIGHT_MATRIX, new Matrix([[0.15, 0.20], [0.25, 0.30]]));
        h.optimizer.setValue(Dense.BIAS_VECTOR, new Vector([0.35]));

        o.optimizer.setValue(Dense.WEIGHT_MATRIX, new Matrix([[0.40, 0.45], [0.50, 0.55]]));
        o.optimizer.setValue(Dense.BIAS_VECTOR, new Vector([0.60]));

        await m.predict([0.05, 0.10]);

        const hOut = h.output.getDefaultValue();
        const oOut = o.output.getDefaultValue();

        hOut.getAt([0]).should.be.closeTo(0.593269, 0.00001);
        hOut.getAt([1]).should.be.closeTo(0.596884, 0.00001);

        oOut.getAt([0]).should.be.closeTo(0.7513650, 0.00001);
        oOut.getAt([1]).should.be.closeTo(0.7729284, 0.00001);
      },
    );


    it(
      'should perform backward pass',
      async () => {
        h.optimizer.setValue(Dense.WEIGHT_MATRIX, new Matrix([[0.15, 0.20], [0.25, 0.30]]));
        h.optimizer.setValue(Dense.BIAS_VECTOR, new Vector([0.35]));

        o.optimizer.setValue(Dense.WEIGHT_MATRIX, new Matrix([[0.40, 0.45], [0.50, 0.55]]));
        o.optimizer.setValue(Dense.BIAS_VECTOR, new Vector([0.60]));

        await m.fit([0.05, 0.10], [0.01, 0.99]);

        const oBack = o.backpropOutput.getValue(Layer.DERIVATIVE);
        const hBack = h.backpropOutput.getValue(Layer.DERIVATIVE);

        oBack.getAt(0).should.be.closeTo(0.186815, 0.00001);

      },
    );
  },
);
