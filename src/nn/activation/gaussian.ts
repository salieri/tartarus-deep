import { Activation } from './activation';
import { Vector } from '../../math';


/**
 * Gaussian
 * @link https://en.wikipedia.org/wiki/Activation_function
 */
export class Gaussian extends Activation {
  public calculate(z: Vector): Vector {
    // e^-(x^2)
    return z.pow(2).neg().exp();
  }


  public derivative(a: Vector, z: Vector): Vector {
    const two = z.set(2);

    // -2xe^(-x^2)
    return two.neg().mul(z).mul(z.pow(2).neg().exp());
  }
}

