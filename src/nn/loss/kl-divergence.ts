import { Loss } from './loss';
import { Vector } from '../../math';


/**
 * Generalized Kullback Leibler Divergence
 * @link https://stats.stackexchange.com/questions/154879/a-list-of-cost-functions-used-in-neural-networks-alongside-applications
 */
export class KlDivergence extends Loss {
  public calculate(yHat: Vector, y: Vector): number {
    // y log (y / yHat)
    return y.log().mul(y.div(yHat)).sum();
  }


  public gradient(yHat: Vector, y: Vector): Vector {
    return y.neg().div(yHat);
  }
}

