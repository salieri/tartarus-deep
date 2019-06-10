import { Activation } from './activation';
import { NDArray, NDArrayPosition, Vector } from '../../math';

/**
 * Softmax
 * @link https://deepnotes.io/softmax-crossentropy
 * @link https://medium.com/@aerinykim/how-to-implement-the-softmax-derivative-independently-from-any-loss-function-ae6d44363a9d
 * @link https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
 * @link https://peterroelants.github.io/posts/cross-entropy-softmax/
 * @link https://stats.stackexchange.com/questions/265905/derivative-of-softmax-with-respect-to-weights
 */
export class Softmax extends Activation {
  public calculate(z: NDArray): NDArray {
    const max = z.max(); // numerically stable softmax
    const exp = z.sub(max).exp();

    return exp.div(exp.sum());
  }


  public derivative(a: NDArray): NDArray {
    const v = new Vector(a);

    const jacobian = v.diagonal();

    const grad = jacobian.apply(
      (val: number, pos: NDArrayPosition) => {
        const i = pos[0];
        const j = pos[1];

        const iVal = v.getAt([i]);

        return (i === j) ? iVal * (1 - iVal) : -iVal * v.getAt([j]);
      },
    );

    return grad.mul(a);
  }
}
