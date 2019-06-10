import { Activation } from './activation';

import {
  Matrix,
  NDArray,
  NDArrayPosition,
  Vector,
} from '../../math';

/**
 * Softmax
 * @link https://deepnotes.io/softmax-crossentropy
 * @link https://medium.com/@aerinykim/how-to-implement-the-softmax-derivative-independently-from-any-loss-function-ae6d44363a9d
 * @link https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
 * @link https://peterroelants.github.io/posts/cross-entropy-softmax/
 * @link https://stats.stackexchange.com/questions/265905/derivative-of-softmax-with-respect-to-weights
 * @link https://math.stackexchange.com/questions/2843505/derivative-of-softmax-without-cross-entropy
 */
export class Softmax extends Activation {
  public calculate(z: NDArray): NDArray {
    const max = z.max(); // numerically stable softmax
    const exp = z.sub(max).exp();

    return exp.div(exp.sum());
  }


  public derivative(a: NDArray, z: NDArray, y: NDArray): NDArray {
    if (!y) {
      throw new Error('Expected training label to be defined');
    }

    const v = new Vector(a);

    const jacobian = v.diagonal();

    const grad = new Matrix(
      jacobian.apply(
        (val: number, pos: NDArrayPosition) => {
          const i = pos[0];
          const j = pos[1];

          const iVal = v.getAt([i]);

          return (i === j) ? iVal * (1 - iVal) : -iVal * v.getAt([j]);
        },
      ),
    );


    const entropyDerivatives = new Vector(y.neg().div(a));

    return grad.vecmul(entropyDerivatives);
  }
}
