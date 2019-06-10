import { Activation } from './activation';
import { NDArray } from '../../math';


/**
 * ArcTan
 * @link https://en.wikipedia.org/wiki/Activation_function
 */
export class ArcTan extends Activation {
  public calculate(z: NDArray): NDArray {
    return z.atan();
  }

  public derivative(a: NDArray, z: NDArray): NDArray {
    const one = z.set(1);

    return one.div(one.add(z.pow(2)));
  }
}

