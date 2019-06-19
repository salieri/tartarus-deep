import { Loss } from './loss';
import { Vector } from '../../math';


export class CrossEntropy extends Loss {
  public calculate(yHat: Vector, y: Vector): number {
    // -sum( y * log( yHat ) + ( 1 - y ) * log( 1 - yHat ) ) / y.size

    const oneMinusY     = y.set(1).sub(y);
    const oneMinusYHat  = yHat.set(1).sub(yHat);

    return -(y.mul(yHat.log()).add(oneMinusY.mul(oneMinusYHat.log()))).mean();
  }


  public gradient(yHat: Vector, y: Vector): Vector {
    const one = yHat.set(1);

    return yHat.sub(y).div(one.sub(yHat).mul(yHat));
  }
}

