import { Activation } from './activation';
import { NDArray } from '../../math';


/**
 * Sinc
 * @link https://en.wikipedia.org/wiki/Activation_function
 */
export class Sinc extends Activation {
  public calculate(z: NDArray): NDArray {
    return z.apply(
      (val: number): number => (((val === 0) || (val === 0.0)) ? 1.0 : Math.sin(val) / val),
    );
  }


  public derivative(a: NDArray, z: NDArray): NDArray {
    return z.apply(
      (val: number) => {
        if (val === 0) {
          return 0;
        }

        return (Math.cos(val) / val) - (Math.sin(val) / (val ** 2));
      },
    );
  }
}
