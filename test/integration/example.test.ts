import { Model, Dense, MemoryInputFeed } from '../../src';

describe(
  'Documentation Example',
  () => {
    it(
      'can run the example described in README',
      async () => {
        // 1. Define a model
        const model = new Model({ optimizer: 'stochastic', loss: 'mean-squared-error' });

        model
          // 4 input nodes
          .input(4)
          // Hidden layer with 5 units and sigmoid activation
          .push(new Dense({ units: 5, activation: 'sigmoid' }))
          // Output layer with 3 units and softmax activation
          .push(new Dense({ units: 3, activation: 'softmax' }));

        await model.compile();

        // 2. Prepare input data with labels
        const feed = new MemoryInputFeed();

        feed
          .add([1, 2, 3, 4], [1, 0, 0]) // input, expected label
          .add([4, 3, 2, 1], [0, 1, 0])
          .add([5, 6, 7, 8], [0, 0, 1]);

        // 3. Train model
        await model.fit(feed, { batchSize: 1, epochs: 100 });

        // 4. Predict
        const result = await model.predict([8, 9, 10, 11]);
      },
    );
  },
);
