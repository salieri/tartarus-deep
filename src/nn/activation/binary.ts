import { Activation } from './activation';
import { NDArray } from '../../math';


/**
 * Binary step
 */
export class Binary extends Activation {
  public calculate(z: NDArray): NDArray {
    return z.apply((val: number): number => (val < 0 ? 0 : 1));
  }
}

