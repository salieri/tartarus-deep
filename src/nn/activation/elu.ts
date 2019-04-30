import Joi from 'joi'; // Can't use JoiEx here -- circular dependency
import { Activation, ActivationDescriptor } from './activation';
import { NDArray } from '../../math';


/**
 * Exponential linear unit
 */
export class ELU extends Activation {
  public calculate(z: NDArray): NDArray {
    return z.apply((val: number): number => (val < 0 ? this.params.leak * (Math.exp(val) - 1) : val));
  }


  public getDescriptor(): ActivationDescriptor {
    return {
      leak: Joi.number().optional().default(0.0).description('Leak multiplier'),
    };
  }
}
