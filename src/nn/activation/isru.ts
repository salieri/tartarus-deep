import Joi from 'joi'; // Can't use JoiEx here -- circular dependency
import { Activation, ActivationParams } from './activation';
import { Vector } from '../../math';

/* eslint-disable @typescript-eslint/interface-name-prefix */
export interface ISRUParams extends ActivationParams {
  alpha?: number;
}


/**
 * Inverse square root unit
 * @link https://en.wikipedia.org/wiki/Activation_function
 */
export class ISRU extends Activation<ISRUParams> {
  public calculate(z: Vector): Vector {
    // z / ( sqrt( 1 + alpha * z^2 )
    return z.div(z.pow(2).mul(this.params.alpha).add(1).sqrt());
  }


  public derivative(a: Vector, z: Vector): Vector {
    const one = z.set(1);

    // ( 1 / sqrt( 1 + alpha * z ^ 2 ) ) ^ 3
    return one.div(one.add(z.pow(2).mul(this.params.alpha))).pow(3);
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

