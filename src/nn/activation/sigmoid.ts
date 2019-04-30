import { Activation } from './activation';
import { NDArray } from '../../math';


/**
 * Sigmoid / soft step / logistic
 */
export class Sigmoid extends Activation {
  public calculate(z: NDArray): NDArray {
    // 1 / ( 1 + e^-z )
    const one = z.set(1.0);

    return one.div(z.neg().exp().add(1));
  }
}
