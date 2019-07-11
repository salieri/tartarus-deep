# Tartarus Deep Learning Framework

**Deep learning framework for TypeScript.** Run it on a browser, on AWS Lambda, or on anything that runs Node.js!


[![Travis CI](https://travis-ci.org/salieri/tartarus-deep.svg?branch=master)](https://travis-ci.org/salieri/tartarus-deep/)
[![Coverage Status](https://coveralls.io/repos/github/salieri/tartarus-deep/badge.svg?branch=master)](https://coveralls.io/github/salieri/tartarus-deep?branch=master)
[![Codecov](https://codecov.io/gh/salieri/tartarus-deep/branch/master/graph/badge.svg)](https://codecov.io/gh/salieri/tartarus-deep)
[![Codacy](https://api.codacy.com/project/badge/Grade/a7f08c24980f47e9b33a791903545fca)](https://www.codacy.com/app/salieri/tartarus-deep?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=salieri/tartarus-deep&amp;utm_campaign=Badge_Grade)
[![Maintainability](https://api.codeclimate.com/v1/badges/8ff95e28307f14562c3c/maintainability)](https://codeclimate.com/github/salieri/tartarus-deep/maintainability)
[![David](https://david-dm.org/salieri/tartarus-deep.svg)](https://david-dm.org/salieri/tartarus-deep)
[![David](https://david-dm.org/salieri/tartarus-deep/dev-status.svg)](https://david-dm.org/salieri/tartarus-deep?type=dev)


## Features

From-the-ground-up implementation for:

*   **Math:** Vector and Matrix operations, seeded randomization
*   **Graph:** Acyclic networks, automated data routing, support for multiple
    input and output layers
*   **Machine Learning:** Forward and back propagation, logistic regression,
    gradient descent, loss (cost) functions, activation functions, optimizers,
    metrics, dense layers, concat layers


## Goals

*   From-the-ground-up implementation for all standard deep learning operations
*   Compatibility with Node.js, modern browsers, and AWS Lambda



## Example

```ts
import { Model, Dense, MemoryInputFeed } from '@tartarus/deep';

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
    }
  );
```

## Model

## Layers

### Dense Layer

### Concat Layer

## Supplementary

### Activation Functions

### Loss Functions

### Initializers

### Metrics

### Input

