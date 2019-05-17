import Joi from 'joi'; // Can't use JoiEx here -- circular dependency
import { Activation, ActivationParams } from './activation';
import { NDArray } from '../../math';


/* eslint-disable @typescript-eslint/interface-name-prefix */
export interface ISRLUParamsInput extends ActivationParams {
  alpha?: number;
}


/**
 * Inverse square root linear unit
 */
export class ISRLU extends Activation<ISRLUParamsInput> {
  public calculate(z: NDArray): NDArray {
    return z.apply(
      (val: number): number => {
        if (val < 0) {
          // 1 / sqrt( 1 + alpha * x^2 )
          return 1.0 / Math.sqrt(1.0 + this.params.alpha * (val ** 2));
        }

        return val;
      },
    );
  }


  public getParamSchema(): Joi.Schema {
    return Joi.object().keys(
      {
        alpha: Joi.number().optional().default(0.0).description('Multiplier'),
      },
    );
  }
}

