import { ConcatSum } from '../../examples';

import {
  DeferredCollection,
  DeferredInputCollection,
  Model,
  Vector,
} from '../../src';
import { shouldSkipSlowTests } from '../util';


// eslint-disable-next-line no-unused-vars
const skipped = shouldSkipSlowTests() ? true : describe(
  'Dense Network Lifecycle for ConcatSum',
  () => {
    const generator = new ConcatSum();

    let model: Model;


    before(
      async () => {
        model = generator.model();

        await model.compile();
      },
    );


    it(
      'should train the model with mini batches',
      async function () {
        this.timeout(10 * 60 * 1000);

        const epochs = 100;
        const sampleCount = 72;
        const batchSize = 1;
        const samples = generator.samples(sampleCount);


        await model.fit(
          samples,
          {
            epochs,
            batchSize,
          },
        );

        const input = new DeferredInputCollection();

        input.set('a', new DeferredCollection(new Vector([12])));
        input.set('b', new DeferredCollection(new Vector([16, 9])));

        const result = await model.predict(input);

        result.getDefaultValue().getAt(0).should.be.closeTo(12 + 16 + 9, 0.01);
      },
    );
  },
);
