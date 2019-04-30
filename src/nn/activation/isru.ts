import { Activation, ActivationDescriptor } from './activation';
import { NDArray } from '../../math';
import Joi from 'joi'; // Can't use JoiEx here -- circular dependency


/**
 * Inverse square root unit
 */
export class ISRU extends Activation {
  public calculate(z: NDArray): NDArray {
    // z / ( sqrt( 1 + alpha * z^2 )
    return z.div(z.pow(2).mul(this.params.alpha).add(1).sqrt());
  }


  public getDescriptor(): ActivationDescriptor {
    return {
      alpha: Joi.number().optional().default(0.0).description('Multiplier')
    };
  }
}

