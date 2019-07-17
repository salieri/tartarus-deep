import Joi from '@hapi/joi'; // Can't use JoiEx here -- circular dependency
import { Activation, ActivationParams } from './activation';
import { Vector } from '../../math';


/* eslint-disable @typescript-eslint/interface-name-prefix */
export interface ISRLUParamsInput extends ActivationParams {
  alpha?: number;
}


/**
 * Inverse square root linear unit
 * @link https://en.wikipedia.org/wiki/Activation_function
 */
export class ISRLU extends Activation<ISRLUParamsInput> {
  public calculate(z: Vector): Vector {
    return z.apply(
      (val: number): number => {
        if (val < 0) {
          // x / sqrt( 1 + alpha * x^2 )
          return val / Math.sqrt(1 + (this.params.alpha * (val ** 2)));
        }

        return val;
      },
    );
  }


  public gradient(a: Vector, z: Vector): Vector {
    return z.apply(
      (val: number): number => {
        if (val >= 0) {
          return 1;
        }

        // (1 / sqrt( 1 + alpha * z^2 ))^3
        return 1 / (Math.sqrt(1 + (this.params.alpha * (val ** 2))) ** 3);
      },
    );
  }


  public getParamSchema(): Joi.Schema {
    return Joi.object().keys(
      {
        alpha: Joi.number().optional().default(0.01).description('Multiplier'),
      },
    );
  }
}

