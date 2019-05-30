import { Dense2x } from '../../examples';
import { Model } from '../../src';


describe(
  'Dense Network Lifecycle',
  () => {
    const generator = new Dense2x();

    let model: Model;


    it(
      'should generate a model that can learn how to multiply a value by 2x',
      async () => {
        model = generator.model();

        await model.compile();
        await model.initialize();
      },
    );


    it.skip(
      'should train the model with 100 samples',
      async () => {
        const data = generator.samples(100);

        await model.fit(data);
      },
    );


    it(
      'should predict a result on input data',
      async () => {
        const testData = generator.samples(20);

        await model.predict(4);
      },
    );


    it(
      'should evaluate the model performance',
      async () => {
        console.log(await model.evaluate(4, 8));
      },
    );
  },
);
