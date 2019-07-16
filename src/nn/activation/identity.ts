import { Activation } from './activation';
import { Vector } from '../../math';


/**
 * Identity
 */
export class Identity extends Activation {
  public calculate(z: Vector): Vector {
    return z;
  }


  public gradient(a: Vector): Vector {
    return a.set(1);
  }
}

