import { Loss } from './loss';
import { Vector } from '../../math';


/**
 * Generalized Kullback Leibler Divergence
 * @link https://stats.stackexchange.com/questions/154879/a-list-of-cost-functions-used-in-neural-networks-alongside-applications
 */
export class GeneralizedKlDivergence extends Loss {
  public calculate(yHat: Vector, y: Vector): number {
    // ( sum( y * log( y ) ) / y.size ) - ( sum( y * log( yHat ) ) / y.size )
    const crossEntropy  = y.mul(yHat.log()).mean();
    const entropy       = y.mul(y.log()).mean();

    return entropy - crossEntropy;
  }


  public gradient(yHat: Vector, y: Vector): Vector {
    return yHat.sub(y).div(yHat);
  }
}
