import { Loss } from './loss';
import { Vector } from '../../math';

/**
 * Mean Squared Logarithmic Error
 */
export class MeanSquaredLogarithmicError extends Loss {
  public calculate(yHat: Vector, y: Vector): number {
    // sum( ( log( y + 1 ) - log( yHat + 1 ) ) ^ 2 ) / y.size

    const yHatPlusOneLog  = yHat.add(1.0).log();
    const yPlusOneLog     = y.add(1.0).log();

    return yPlusOneLog.sub(yHatPlusOneLog).pow(2).mean();
  }

  public gradient(yHat: Vector, y: Vector): Vector {
    return y.zero(); // UNFINISHED
  }
}

