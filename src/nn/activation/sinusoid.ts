import { Activation } from './activation';
import { NDArray } from '../../math';


/**
 * Sinusoid
 * @link https://en.wikipedia.org/wiki/Activation_function
 */
export class Sinusoid extends Activation {
  public calculate(z: NDArray): NDArray {
    return z.sin();
  }


  public derivative(a: NDArray, z: NDArray): NDArray {
    return z.cos();
  }
}
