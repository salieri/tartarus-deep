import { Activation } from './activation';
import { Vector } from '../../math';
import { Sigmoid } from './sigmoid';


/**
 * Sigmoid-weighted linear unit / SiLU / Swish-1
 * @link https://en.wikipedia.org/wiki/Activation_function
 */
export class SiLU extends Activation {
  private static sigmoid = new Sigmoid();

  public calculate(z: Vector): Vector {
    return z.mul(SiLU.sigmoid.calculate(z));
  }


  /**
   * @param a activated value (result of this.calculate(z))
   * @param z linear value
   */
  public derivative(a: Vector, z: Vector): Vector {
    const one = a.set(1);

    return a.add(SiLU.sigmoid.calculate(z).mul(one.sub(a)));
  }
}
