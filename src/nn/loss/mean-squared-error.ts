import { Loss } from './loss';
import { Vector } from '../../math';

/**
 * Mean Squared Error
 * A.k.a. Quadratic Loss
 * @link https://ml-cheatsheet.readthedocs.io/en/latest/calculus.html
 */
export class MeanSquaredError extends Loss {
  public calculate(yHat: Vector, y: Vector): number {
    // 0.5 * sum( ( yHat - y ) ^ 2 ) / y.size
    return 0.5 * yHat.sub(y).pow(2).mean();
  }


  public gradient(yHat: Vector, y: Vector): Vector {
    return yHat.sub(y);
  }
}

