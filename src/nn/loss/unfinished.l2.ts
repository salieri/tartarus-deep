import { Loss } from './loss';
import { Vector } from '../../math';

/**
 * L2
 */
export class L2 extends Loss {
  public calculate(yHat: Vector, y: Vector): number {
    // sum( ( yHat - y ) ^ 2 )
    return yHat.sub(y).pow(2).sum();
  }

  public gradient(yHat: Vector, y: Vector): Vector {
    return y.zero(); // UNFINISHED
  }

  // public gradient(yHat: Vector, y: Vector, x: Vector): Vector {
  //   return x.mul(2);
  // }
}
