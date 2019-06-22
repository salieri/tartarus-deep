# Tartarus Deep Learning Framework

**Deep learning framework for TypeScript.** Run it on a browser, on AWS Lambda, or on anything that runs Node.js!


[![Travis CI](https://travis-ci.org/franksrevenge/tartarus-deep.svg?branch=master)](https://travis-ci.org/franksrevenge/tartarus-deep/)
[![Coverage Status](https://coveralls.io/repos/github/franksrevenge/tartarus-deep/badge.svg?branch=master)](https://coveralls.io/github/franksrevenge/tartarus-deep?branch=master)
[![Codacy](https://api.codacy.com/project/badge/Grade/8279d1926eed411cae160fc6c9156560)](https://www.codacy.com/app/franksrevenge/tartarus-deep?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=franksrevenge/tartarus-deep&amp;utm_campaign=Badge_Grade)
[![Maintainability](https://api.codeclimate.com/v1/badges/d44d858f3266337623c8/maintainability)](https://codeclimate.com/github/franksrevenge/tartarus-deep/maintainability)
[![David](https://david-dm.org/franksrevenge/tartarus-deep.svg)](https://david-dm.org/franksrevenge/tartarus-deep)
[![David](https://david-dm.org/franksrevenge/tartarus-deep/dev-status.svg)](https://david-dm.org/franksrevenge/tartarus-deep?type=dev)


## Platform Goals

### 1. From-The-Ground-Up Implementation

*   Vector operations ✔
*   Matrix operations ✔
*   Backpropagation ✔
*   Forward propagation ✔
*   Optimization ✔
*   Logistic regression ✔
*   Gradient descent ✔
*   Loss functions ✔
*   Activation functions ✔
*   Cost functions ✔
*   Neural networks ✔
*   Acyclic networks ✔
*   CNNs
*   RNNs
*   Regularization
*   Graph networks ✔
*   Chainable models ✔
*   Randomization ✔
*   Metrics ⏳
*   Optimizers ⏳



### 2. Compatibility


*   Runs on Node.js ✔
*   Runs on modern browsers
*   Runs on AWS Lambda


### 3. Speed

*   Decidedly unoptimized -- written for research purposes, not for speed.



## Acknowledgements

Many ideas, designs, algorithms, and approaches have been shamelessly stolen from:

*   <https://isaacchanghau.github.io/post/loss_functions/>
*   <https://ml-cheatsheet.readthedocs.io/en/latest/>
*   <https://en.wikipedia.org/wiki/Activation_function>
*   <https://keras.io/>
*   <https://www.tensorflow.org/api_docs/python/>
*   <https://mathinsight.org/matrix_vector_multiplication>
*   <https://stackoverflow.com/a/47593316/844771>
*   <http://www.numpy.org/>
*   <https://www.coursera.org/specializations/deep-learning>
*   <https://www.quora.com/How-does-Keras-calculate-accuracy>
*   <https://brilliant.org/wiki/backpropagation/>
*   <https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/>
*   <https://eli.thegreenplace.net/2018/backpropagation-through-a-fully-connected-layer/>
*   <https://towardsdatascience.com/recurrent-neural-networks-and-lstm-4b601dd822a5>
*   <https://theclevermachine.wordpress.com/2014/09/08/derivation-derivatives-for-common-neural-network-activation-functions/>
*   <https://sefiks.com/category/machine-learning/>
*   <https://deepnotes.io/softmax-crossentropy>
*   <https://cup-of-char.com/writing-activation-functions-from-mostly-scratch-in-python/>
*   <https://www.youtube.com/watch?v=PPLop4L2eGk&list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN>
*   <https://www.anotsorandomwalk.com/backpropagation-example-with-numbers-step-by-step/>


## Glossary

| Term                 | Description                                            |
| :------------------- | :------------------------------------------------------|
| `a` | Activation |
| `b` | Bias |
| `c` |  |
| `w` | Weight |
| `x` | Input |
| `yHat` | Predicted output
| `<t>` | Time _t_ denotation |
| `[l]` | Layer _l_ denotation |




## Quality

*   TypeScript: ESLint & TSLint with Airbnb-like presets
*   Markdown: Remark Lint with recommended presets
