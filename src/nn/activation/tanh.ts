import { Activation } from './activation';
import { NDArray } from '../../math';


/**
 * TanH
 * @link https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html
 */
export class TanH extends Activation {
  public calculate(z: NDArray): NDArray {
    // (e^z - e^-z) / (e^z + e^-z)
    return z.exp().sub(z.neg().exp()).div(z.exp().add(z.neg().exp()));
  }


  public derivative(a: NDArray): NDArray {
    const one = a.set(1);

    // tanh'(z) = 1 - tanh(z)^2 = 1 - a^2
    return one.sub(a.pow(2));
  }
}
