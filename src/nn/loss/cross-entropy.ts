import { Loss } from './loss';
import { Vector } from '../../math';


export class CrossEntropy extends Loss {
  calculate(yHat: Vector, y: Vector): number {
    // -sum( y * log( yHat ) + ( 1 - y ) * log( 1 - yHat ) ) / y.size

    const oneMinusY     = y.clone().set(1).sub(y);
    const oneMinusYHat  = yHat.clone().set(1).sub(yHat);

    return -(y.mul(yHat.log()).add(oneMinusY.mul(oneMinusYHat.log()))).mean();
  }
}

