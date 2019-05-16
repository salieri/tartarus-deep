import Joi from 'joi'; // Can't use JoiEx here -- circular dependency
import { Activation, ActivationParams } from './activation';
import { NDArray } from '../../math';


export interface ReLUParams extends ActivationParams {
  leak?: number;
}


/**
 * Rectified linear unit / leaky rectified linear unit / parameteric rectified linear unit
 */
export class ReLU extends Activation<ReLUParams> {
  private readonly leak: number;

  public constructor(params: ReLUParams = {}) {
    super(params);

    this.leak = params.leak || 0;
  }


  public calculate(z: NDArray): NDArray {
    return z.apply((val: number): number => (val < 0 ? this.leak * val : val));
  }


  public getParamSchema(): Joi.Schema {
    return Joi.object().keys(
      {
        leak: Joi.number().optional().default(0.0).description('Leak multiplier'),
      },
    );
  }
}
