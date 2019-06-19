import { Loss } from './loss';
import { Vector } from '../../math';

/**
 * Mean Squared Error
 * A.k.a. Quadratic Loss
 */
export class MeanSquaredError extends Loss {
  public calculate(yHat: Vector, y: Vector): number {
    // sum( ( yHat - y ) ^ 2 ) / y.size
    return yHat.sub(y).pow(2).mean();
  }


  public gradient(yHat: Vector, y: Vector): Vector {
    return yHat.sub(y);
  }
}

