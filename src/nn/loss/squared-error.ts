import { Loss } from './loss';
import { Vector } from '../../math';

/**
 * Squared Error
 */
export class SquaredError extends Loss {
  public calculate(yHat: Vector, y: Vector): number {
    // 0.5 * sum( ( yHat - y ) ^ 2 )
    return 0.5 * yHat.sub(y).pow(2).sum();
  }


  public gradient(yHat: Vector, y: Vector): Vector {
    return yHat.sub(y).pow(2).mul(0.5);
    // return yHat.sub(y);
  }
}

