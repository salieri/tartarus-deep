import { Activation } from './activation';
import { Vector } from '../../math';


/**
 * Sinusoid
 * @link https://en.wikipedia.org/wiki/Activation_function
 */
export class Sinusoid extends Activation {
  public calculate(z: Vector): Vector {
    return z.sin();
  }


  public derivative(a: Vector, z: Vector): Vector {
    return z.cos();
  }
}
