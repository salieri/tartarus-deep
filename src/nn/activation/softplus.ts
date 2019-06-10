import { Activation } from './activation';
import { NDArray } from '../../math';


/**
 * SoftPlus
 * @link https://en.wikipedia.org/wiki/Activation_function
 */
export class Softplus extends Activation {
  public calculate(z: NDArray): NDArray {
    // log( 1 + e^z )
    return z.exp().add(1).log();
  }


  public derivative(a: NDArray, z: NDArray): NDArray {
    const one = z.set(1);

    // 1 / 1 + e^(-z)
    return one.div(one.add(z.neg().exp()));
  }
}
