import { DenseSimple2x } from '../../examples';
import { Model } from '../../src';


describe(
  'Dense Network Lifecycle for DenseSimple2x',
  () => {
    const generator = new DenseSimple2x();

    let model: Model;


    before(
      async () => {
        model = generator.model();

        await model.compile();
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

        fitResult.iterations.should.equal(epochs * sampleCount);
        fitResult.batches.should.equal((epochs * sampleCount) / batchSize);
        fitResult.epochs.should.equal(epochs);

        for (let i = 100; i < 1000; i += 15) {
          /* eslint-disable-next-line no-await-in-loop */
          const result = await model.predict(i);

          result.getDefaultValue().getAt(0).should.be.closeTo(i * 2, 0.000000001);
        }
      },
    );
  },
);
