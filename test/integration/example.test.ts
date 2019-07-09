import { Model, Dense, MemoryInputFeed } from '../../src';

describe(
  'Documentation Example',
  () => {
    it(
      'can run the example described in README',
      async () => {
        /*
         * 1. Define a model
         *    - 4 input nodes
         *    - hidden layer with 5 nodes and sigmoid activation
         *    - output layer with 3 nodes and softmax activation
         */
        const model = new Model({ optimizer: 'stochastic', loss: 'mean-squared-error' });

        model
          .input(4)
          .push(new Dense({ units: 5, activation: 'sigmoid' }))
          .push(new Dense({ units: 3, activation: 'softmax' }));

        model.compile()
          .then(
            async () => {
              /* 2. Prepare three samples of training data */
              const feed = new MemoryInputFeed();

              feed
                .add([1, 2, 3, 4], [1, 0, 0]) // .add(input, expected output)
                .add([4, 3, 2, 1], [0, 1, 0])
                .add([5, 6, 7, 8], [0, 0, 1]);


              /* 3. Train model */
              await model.fit(feed, { batchSize: 1, epochs: 100 });


              /* 4. Predict */
              const result = await model.predict([8, 9, 10, 11]);

              console.log(`Prediction: ${result.getDefaultValue().toJSON()}`);
            },
          );
      },
    );
  },
);
