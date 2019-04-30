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
}
