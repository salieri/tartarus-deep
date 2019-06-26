import { ConcatSum } from '../../examples';

import {
  DeferredCollection,
  DeferredInputCollection,
  Model,
  Vector,
} from '../../src';


/**
 * Works, but slow
 * Skipping since it doesn't do anything different from dense-simple-2x.test
 */
describe.only(
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
      async () => {
        const epochs = 10;
        const sampleCount = 64;
        const batchSize = 8;
        const samples = generator.samples(sampleCount);

        const fitResult = await model.fit(
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

        result.getDefaultValue().getAt(0).should.be.closeTo(12 + 16 + 9, 1);
      },
    );
  },
);
