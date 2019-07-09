import { Loss } from './loss';
import { Vector } from '../../math';

/**
 * Poisson
 */
export class Poisson extends Loss {
  public calculate(yHat: Vector, y: Vector): number {
    // sum( yHat - y * log( yHat ) ) / y.size
    return yHat.sub(y.mul(yHat.log())).mean();
  }

  public gradient(yHat: Vector, y: Vector): Vector {
    return y.zero(); // UNFINISHED
  }
}
