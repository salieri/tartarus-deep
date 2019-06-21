import { Dense, Layer, Model } from '../../src/nn';
import { Matrix, Vector } from '../../src/math';
import { Stochastic } from '../../src/nn/optimizer';

function prepareWeights(h: Dense, o: Dense): void {
  h.data.optimizer.setValue(Dense.WEIGHT_MATRIX, new Matrix([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]));
  h.data.optimizer.setValue(Dense.BIAS_VECTOR, new Vector([0.5]));

  o.data.optimizer.setValue(Dense.WEIGHT_MATRIX, new Matrix([[0.7, 0.9], [0.8, 0.1]]));
  o.data.optimizer.setValue(Dense.BIAS_VECTOR, new Vector([0.5]));
}

/**
 * @link https://www.anotsorandomwalk.com/backpropagation-example-with-numbers-step-by-step/
 */
describe(
  'Dense Model with Known (Precalculated) Data 2',
  () => {
    const m = new Model();
    const optimizer = new Stochastic({ rate: 0.01 });

    const h = new Dense({ units: 2, activation: 'sigmoid', weightOptimizer: optimizer, biasOptimizer: optimizer }, 'hidden');
    const o = new Dense({ units: 2, activation: 'sigmoid', weightOptimizer: optimizer, biasOptimizer: optimizer }, 'output');

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
        prepareWeights(h, o);

        await m.predict([1, 4, 5]);

        const hOut = h.data.output.getDefaultValue();
        const oOut = o.data.output.getDefaultValue();

        hOut.getAt([0]).should.be.closeTo(0.9866, 0.0001);
        hOut.getAt([1]).should.be.closeTo(0.9950, 0.0001);

        oOut.getAt([0]).should.be.closeTo(0.8896, 0.0001);
        oOut.getAt([1]).should.be.closeTo(0.8004, 0.0001);
      },
    );


    it(
      'should calculate error terms during backward pass',
      async () => {
        prepareWeights(h, o);

        await m.fit([1, 4, 5], [0.1, 0.05]);

        const oBack = o.data.backpropOutput.getValue(Layer.ERROR_TERM);
        const hBack = h.data.backpropOutput.getValue(Layer.ERROR_TERM);

        oBack.getDims().should.deep.equal([2]);
        oBack.getAt(0).should.be.closeTo(0.0775735, 0.00001);
        oBack.getAt(1).should.be.closeTo(0.1198838, 0.00001);

        hBack.getDims().should.deep.equal([2]);
        hBack.getAt(0).should.be.closeTo(0.00198264, 0.00001);
        hBack.getAt(1).should.be.closeTo(0.00040082, 0.00001);
      },
    );


    it(
      'should calculate weight and bias derivatives using error terms',
      async () => {
        prepareWeights(h, o);

        await m.fit([1, 4, 5], [0.1, 0.05]);

        const oBack = o.data.backpropOutput.getValue(Layer.ERROR_TERM);
        const hBack = h.data.backpropOutput.getValue(Layer.ERROR_TERM);

        // @ts-ignore
        const odWeight = o.calculateLinearWeightDerivative(new Vector(oBack));
        // @ts-ignore
        const odBias = o.calculateLinearBiasDerivative(new Vector(oBack));

        odWeight.getDims().should.deep.equal([2, 2]);
        odWeight.getAt([0, 0]).should.be.closeTo(0.0765, 0.0001);
        odWeight.getAt([0, 1]).should.be.closeTo(0.0772, 0.0001);
        odWeight.getAt([1, 0]).should.be.closeTo(0.1183, 0.0001);
        odWeight.getAt([1, 1]).should.be.closeTo(0.1193, 0.0001);
        odBias.getAt(0).should.be.closeTo(0.1975, 0.001);

        // @ts-ignore
        const hdWeight = h.calculateLinearWeightDerivative(new Vector(hBack));
        // @ts-ignore
        const hdBias = h.calculateLinearBiasDerivative(new Vector(hBack));

        hdWeight.getDims().should.deep.equal([2, 3]);
        hdWeight.getAt([0, 0]).should.be.closeTo(0.002, 0.0001);
        hdWeight.getAt([0, 1]).should.be.closeTo(0.0079, 0.0001);
        hdWeight.getAt([0, 2]).should.be.closeTo(0.0099, 0.0001);
        hdWeight.getAt([1, 0]).should.be.closeTo(0.0004, 0.0001);
        hdWeight.getAt([1, 1]).should.be.closeTo(0.0016, 0.0001);
        hdWeight.getAt([1, 2]).should.be.closeTo(0.0020, 0.0001);

        // Suspect that this value is wrong in the source document
        // hdBias.getAt(0).should.be.closeTo(0.0008, 0.0001);
        hdBias.getAt(0).should.be.closeTo(0.0009919, 0.0001);
      },
    );


    it(
      'should optimize weights and bias',
      async () => {
        prepareWeights(h, o);

        await m.fit([1, 4, 5], [0.1, 0.05]);

        const oWeight = o.data.optimizer.getValue(Dense.WEIGHT_MATRIX);
        const hWeight = h.data.optimizer.getValue(Dense.WEIGHT_MATRIX);
        const oBias = o.data.optimizer.getValue(Dense.BIAS_VECTOR);
        const hBias = h.data.optimizer.getValue(Dense.BIAS_VECTOR);

        hWeight.getAt([0, 0]).should.be.closeTo(0.1000, 0.0001);
        hWeight.getAt([0, 1]).should.be.closeTo(0.2999, 0.0001);
        hWeight.getAt([0, 2]).should.be.closeTo(0.4999, 0.0001);
        hWeight.getAt([1, 0]).should.be.closeTo(0.2000, 0.0001);
        hWeight.getAt([1, 1]).should.be.closeTo(0.4000, 0.0001);
        hWeight.getAt([1, 2]).should.be.closeTo(0.6000, 0.0001);

        hWeight.getAt([0, 0]).should.be.closeTo(0.1000, 0.0001);
        hWeight.getAt([0, 1]).should.be.closeTo(0.2999, 0.0001);
        hWeight.getAt([0, 2]).should.be.closeTo(0.4999, 0.0001);
        hWeight.getAt([1, 0]).should.be.closeTo(0.2000, 0.0001);
        hWeight.getAt([1, 1]).should.be.closeTo(0.4000, 0.0001);
        hWeight.getAt([1, 2]).should.be.closeTo(0.6000, 0.0001);


        oWeight.getAt([0, 0]).should.be.closeTo(0.6992, 0.0001);
        oWeight.getAt([1, 0]).should.be.closeTo(0.7988, 0.0001);
        oWeight.getAt([0, 1]).should.be.closeTo(0.8992, 0.0001);
        oWeight.getAt([1, 1]).should.be.closeTo(0.0988, 0.0001);

        hBias.getAt(0).should.be.closeTo(0.5000, 0.0001);
        oBias.getAt(0).should.be.closeTo(0.4980, 0.0001);

        // for (let a = 0; a < 20000; a++) {
        //   await m.fit([1, 4, 5], [0.1, 0.05]);
        //
        //   console.log((await m.predict([1,4,5])).getDefaultValue().data);
        //
        // }
        //
        // const mm = 123;
      },
    );
  },
);
