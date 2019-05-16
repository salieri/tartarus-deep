import Joi from 'joi'; // Can't use JoiEx here -- circular dependency
import { Activation, ActivationParams } from './activation';
import { NDArray } from '../../math';


export interface ELUParams extends ActivationParams {
  leak?: number;
}

/**
 * Exponential linear unit
 */
export class ELU extends Activation<ELUParams> {
  private readonly leak: number;

  public constructor(params: ELUParams = {}) {
    super(params);

    this.leak = params.leak || 0;
  }


  public calculate(z: NDArray): NDArray {
    return z.apply((val: number): number => (val < 0 ? this.leak * (Math.exp(val) - 1) : val));
  }


  public getParamSchema(): Joi.Schema {
    return Joi.object().keys(
      {
        leak: Joi.number().optional().default(0.0).description('Leak multiplier'),
      },
    );
  }
}
