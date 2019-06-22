import { Dense, Layer, Model } from '../../src/nn';
import { Matrix, Vector } from '../../src/math';
import { Stochastic } from '../../src/nn/optimizer';

function initWeights(h: Dense, o: Dense): void {
  h.data.optimizer.setValue(Dense.WEIGHT_MATRIX, new Matrix([[0.15, 0.20], [0.25, 0.30]]));
  h.data.optimizer.setValue(Dense.BIAS_VECTOR, new Vector([0.35]));

  o.data.optimizer.setValue(Dense.WEIGHT_MATRIX, new Matrix([[0.40, 0.45], [0.50, 0.55]]));
  o.data.optimizer.setValue(Dense.BIAS_VECTOR, new Vector([0.60]));
}


/**
 * @link https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
 */
describe(
  'Dense Model with Known (Precalculated) Data',
  () => {
    const optimizer = new Stochastic({ rate: 0.5 });
    const m = new Model({ optimizer });

    const h = new Dense({ units: 2, activation: 'sigmoid' });
    const o = new Dense({ units: 2, activation: 'sigmoid' });

    before(
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
        initWeights(h, o);

        await m.predict([0.05, 0.10]);

        const hOut = h.data.output.getDefaultValue();
        const oOut = o.data.output.getDefaultValue();

        hOut.getAt([0]).should.be.closeTo(0.593269, 0.00001);
        hOut.getAt([1]).should.be.closeTo(0.596884, 0.00001);

        oOut.getAt([0]).should.be.closeTo(0.7513650, 0.00001);
        oOut.getAt([1]).should.be.closeTo(0.7729284, 0.00001);
      },
    );


    it(
      'should perform an iteration and update weights',
      async () => {
        initWeights(h, o);

        const input = Model.coerceData([0.05, 0.10]);
        const labels = Model.coerceData([0.01, 0.99]);

        await m.iterate(input, labels);

        const oBack = o.data.backpropOutput.getValue(Layer.ERROR_TERM);
        const hBack = h.data.backpropOutput.getValue(Layer.ERROR_TERM);

        oBack.getAt(0).should.be.closeTo(0.1384985, 0.00001);

        // @ts-ignore
        const dEOverA = h.calculateActivationErrorDerivativeFromChain();

        dEOverA.getAt(0).should.be.closeTo(0.03635036, 0.00001);

        // @ts-ignore
        const odWeight = o.calculateLinearWeightDerivative(new Vector(oBack));

        odWeight.getDims().should.deep.equal([2, 2]);
        odWeight.getAt([0, 0]).should.be.closeTo(0.082167041, 0.000001);

        // @ts-ignore
        const hdWeight = h.calculateLinearWeightDerivative(new Vector(hBack));

        hdWeight.getDims().should.deep.equal([2, 2]);
        hdWeight.getAt([0, 0]).should.be.closeTo(0.000438568, 0.0000001);

        odWeight.equals(o.data.fitter.getValue(Dense.WEIGHT_ERROR)).should.equal(true);
        hdWeight.equals(h.data.fitter.getValue(Dense.WEIGHT_ERROR)).should.equal(true);

        await m.optimize();

        // @ts-ignore
        const hWeight = h.data.optimizer.getValue(Dense.WEIGHT_MATRIX);

        // @ts-ignore
        const oWeight = o.data.optimizer.getValue(Dense.WEIGHT_MATRIX);

        oWeight.getAt([0, 0]).should.be.closeTo(0.35891648, 0.0000001);
        oWeight.getAt([0, 1]).should.be.closeTo(0.40866618, 0.0000001);
        oWeight.getAt([1, 0]).should.be.closeTo(0.51130127, 0.0000001);
        oWeight.getAt([1, 1]).should.be.closeTo(0.56137012, 0.0000001);

        hWeight.getAt([0, 0]).should.be.closeTo(0.14978071, 0.0000001);
        hWeight.getAt([0, 1]).should.be.closeTo(0.19956143, 0.0000001);
        hWeight.getAt([1, 0]).should.be.closeTo(0.24975114, 0.0000001);
        hWeight.getAt([1, 1]).should.be.closeTo(0.29950229, 0.0000001);
      },
    );
  },
);
