import { Loss } from './loss';
import { Vector } from '../../math';

/**
 * Negative Logarithmic Likelihood
 */
export class NegativeLogarithmicLikelihood extends Loss {
  public calculate(yHat: Vector /* , y: Vector */): number {
    // -sum( log( yHat ) ) / y.size
    return -yHat.log().mean();
  }
}

