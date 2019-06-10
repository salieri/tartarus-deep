import { Activation } from './activation';
import { NDArray } from '../../math';


/**
 * Identity
 */
export class Identity extends Activation {
  public calculate(z: NDArray): NDArray {
    return z;
  }


  public derivative(a: NDArray): NDArray {
    return a.set(1);
  }
}

