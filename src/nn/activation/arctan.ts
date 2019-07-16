import { Activation } from './activation';
import { Vector } from '../../math';


/**
 * ArcTan
 * @link https://en.wikipedia.org/wiki/Activation_function
 */
export class ArcTan extends Activation {
  public calculate(z: Vector): Vector {
    return z.atan();
  }

  public gradient(a: Vector, z: Vector): Vector {
    const one = z.set(1);

    return one.div(one.add(z.pow(2)));
  }
}

