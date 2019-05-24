import { Dense, Session } from '../../../../src/nn';
import { LayerOutputUtil } from '../../../util';

describe(
  'Layer: Dense',
  () => {
    it(
      'should reject multiple inputs',
      async () => {
        const data = LayerOutputUtil.createManyOutputs();
        const dense = new Dense({ units: 1 });

        dense.setRawInputs(data);

        await dense.compile().should.be.rejectedWith(/Too many inputs for a dense layer/);
      },
    );


    it(
      'should fail to compile if no inputs are defined',
      async () => {
        const dense = new Dense({ units: 1 });

        await dense.compile().should.be.rejectedWith(/Missing input for dense layer/);
      },
    );


    it(
      'should initialize weight and bias values',
      async () => {
        const dense = new Dense(
          {
            units: 2,
            bias: true,
            biasInitializer: 'zero',
            weightInitializer: 'one',
            activation: 'relu',
          },
        );

        const data = LayerOutputUtil.createOutput();
        const session = new Session('hello-world');

        dense.setSession(session);
        dense.setRawInputs(data);

        await dense.compile();
        await dense.initialize();

        const optimizer = dense.getOptimizer();

        const bias = optimizer.getValue('bias');
        const weight = optimizer.getValue('weight');

        bias.countElements().should.equal(2);
        weight.countElements().should.equal(2 * data.getDefault().getDefault().countElements());

        bias.getAt([0]).should.equal(0);
        weight.getAt([0, 0]).should.equal(1);
      },
    );


    it(
      'should calculate forward propagation',
      async () => {
        const dense = new Dense(
          {
            units: 2,
            bias: true,
            biasInitializer: 'one',
            weightInitializer: 'one',
            activation: 'arctan',
          },
        );

        const data = LayerOutputUtil.createOutput();
        const session = new Session('hello-world');

        dense.setSession(session);
        dense.setRawInputs(data);

        await dense.compile();
        await dense.initialize();
        await dense.forward();

        const out = dense.getRawOutputs().getDefault();

        const linear = out.get('linear').get();
        const activated = out.get('activated').get();

        activated.should.equal(out.getDefault().get());

        linear.getAt([0]).should.equal((4 * 1 + 5 * 1 + 6 * 1 + 7 * 1) + 1);

        activated.getAt([0]).should.equal(Math.atan(linear.getAt([0])));
      },
    );
  },
);

