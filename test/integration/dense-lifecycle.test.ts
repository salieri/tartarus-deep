import { Dense2x } from '../../examples';
import { Dense } from '../../src';


describe(
  'Dense Network Lifecycle',
  () => {
    const generator = new Dense2x();

    let model;


    it(
      'should generate a model that can learn how to multiply a value by 2x',
      () => {
        model = generator.model();
      },
    );


    it(
      'should train the model with 100 samples',
      async () => {
        const data = generator.samples(100);

        await model.fit(data);
      },
    );


    it(
      'should evaluate the model performance',
      async () => {
        await model.evaluate();
      },
    );


    it(
      'should predict',
      async () => {
        const testData = generator.samples(20);

        await model.predict(testData);
      },
    );
  },
);
