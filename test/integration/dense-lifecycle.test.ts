import { Dense2x } from '../../examples';
import { Model } from '../../src';


/**
 * Works, but slow
 * Skipping since it doesn't do anything different from dense-simple-2x.test
 */
describe.skip(
  'Dense Network Lifecycle for Dense2x',
  () => {
    const generator = new Dense2x();

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
        const epochs = 200;
        const sampleCount = 128;
        const batchSize = 16;
        const samples = generator.samples(sampleCount);

        const fitResult = await model.fit(
          samples,
          {
            epochs,
            batchSize,
          },
        );

        fitResult.iterations.should.equal(epochs * sampleCount);
        fitResult.batches.should.equal((epochs * sampleCount) / batchSize);
        fitResult.epochs.should.equal(epochs);

        for (let i = 100; i < 1000; i += 15) {
          /* eslint-disable-next-line no-await-in-loop */
          const result = await model.predict(i);

          result.getDefaultValue().getAt(0).should.be.closeTo(i * 2, 1);
        }
      },
    );
  },
);
