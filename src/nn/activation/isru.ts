import Joi from 'joi'; // Can't use JoiEx here -- circular dependency
import { Activation, ActivationParams } from './activation';
import { NDArray } from '../../math';

/* eslint-disable @typescript-eslint/interface-name-prefix */
export interface ISRUParams extends ActivationParams {
  alpha?: number;
}


/**
 * Inverse square root unit
 */
export class ISRU extends Activation<ISRUParams> {
  protected readonly alpha: number;

  public constructor(params: ISRUParams = {}) {
    super(params);

    this.alpha = params.alpha || 0;
  }


  public calculate(z: NDArray): NDArray {
    // z / ( sqrt( 1 + alpha * z^2 )
    return z.div(z.pow(2).mul(this.alpha).add(1).sqrt());
  }


  public getParamSchema(): Joi.Schema {
    return Joi.object().keys(
      {
        alpha: Joi.number()
          .optional()
          .default(0.0)
          .description('Multiplier'),
      },
    );
  }
}

