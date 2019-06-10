import { Activation } from './activation';
import { NDArray } from '../../math';


/**
 * Bent identity
 * @link https://en.wikipedia.org/wiki/Activation_function
 */
export class BentIdentity extends Activation {
  public calculate(z: NDArray): NDArray {
    // ( sqrt( z^2 + 1 ) - 1 ) / 2 + z
    return z.pow(2).add(1).sqrt().sub(1)
      .div(2)
      .add(z);
  }


  public derivative(a: NDArray, z: NDArray): NDArray {
    const one = z.set(1);
    const two = z.set(2);

    // (x / [2 * sqrt(x^2+1))] + 1
    return z.div(two.mul(z.pow(2).add(one).sqrt())).add(one);
  }
}
