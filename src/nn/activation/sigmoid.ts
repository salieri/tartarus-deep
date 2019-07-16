import { Activation } from './activation';
import { Vector } from '../../math';


/**
 * Sigmoid / soft step / logistic
 * @link https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html
 */
export class Sigmoid extends Activation {
  public calculate(z: Vector): Vector {
    // S(z) = 1 / ( 1 + e^(-z) )
    const one = z.set(1.0);

    return one.div(z.neg().exp().add(1));
  }


  public gradient(a: Vector): Vector {
    const one = a.set(1);

    // S'(z) = S(z) * (1 - S(z))
    return a.mul(one.sub(a));
  }
}
