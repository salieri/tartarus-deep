import { Activation } from './activation';
import { NDArray } from '../../math';


/**
 * Softsign
 * @link https://en.wikipedia.org/wiki/Activation_function
 */
export class Softsign extends Activation {
  public calculate(z: NDArray): NDArray {
    // z / ( 1 + |z| )
    return z.div(z.abs().add(1.0));
  }


  public derivative(a: NDArray, z: NDArray): NDArray {
    const one = z.set(1);

    // 1 / ((1 + |z|)^2)
    return one.div(one.add(z.abs()).pow(2));
  }
}
