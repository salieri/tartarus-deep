import Joi from 'joi'; // Can't use JoiEx here -- circular dependency
import { Activation, ActivationParams } from './activation';
import { NDArray } from '../../math';


export interface ELUParamsInput extends ActivationParams {
  leak?: number;
}

export interface ELUParamsCoerced extends ActivationParams {
  leak: number;
}


/**
 * Exponential linear unit
 */
export class ELU extends Activation<ELUParamsInput, ELUParamsCoerced> {
  public calculate(z: NDArray): NDArray {
    return z.apply((val: number): number => (val < 0 ? this.params.leak * (Math.exp(val) - 1) : val));
  }


  public getParamSchema(): Joi.Schema {
    return Joi.object().keys(
      {
        leak: Joi.number().optional().default(0.0).description('Leak multiplier'),
      },
    );
  }
}
