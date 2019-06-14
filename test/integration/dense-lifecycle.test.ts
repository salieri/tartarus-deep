import { Dense2x } from '../../examples';
import { Model } from '../../src';


describe.only(
  'Dense Network Lifecycle',
  () => {
    const generator = new Dense2x();

    let model: Model;


    it(
      'should generate a model which can learn how to multiply a value by 2x',
      async () => {
        model = generator.model();

        await model.compile();
        await model.initialize();
      },
    );


    it(
      'should predict a result on input data',
      async () => {
        const result = await model.predict(4);
        const result2 = await model.predict(2);

        const nd = result.getDefaultValue();
        const nd2 = result2.getDefaultValue();

        nd.countDims().should.equal(1);
        nd2.countDims().should.equal(1);

        nd.countElements().should.equal(1);
        nd.countElements().should.equal(1);

        nd.getId().should.not.equal(nd2.getId());
        nd.equals(nd2).should.not.equal(true);
      },
    );


    it(
      'should evaluate model performance',
      async () => {
        const result = await model.evaluate(4, 8);

        const nd = result.losses.getDefaultValue();

        nd.countDims().should.equal(1);
        nd.countElements().should.equal(1);
      },
    );


    it(
      'should train the model with 100 samples',
      async () => {
        // const data = generator.samples(100);

        await model.fit(4, 8);
      },
    );
  },
);
