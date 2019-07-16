import Joi from 'joi'; // Can't use JoiEx here -- circular dependency
import { Activation, ActivationParams } from './activation';
import { Vector } from '../../math';


export interface ReLUParams extends ActivationParams {
  leak?: number;
}


/**
 * Rectified linear unit / leaky rectified linear unit / parameteric rectified linear unit
 * @link https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html
 * @link https://sefiks.com/2018/02/26/leaky-relu-as-an-neural-networks-activation-function/
 */
export class ReLU extends Activation<ReLUParams> {
  public calculate(z: Vector): Vector {
    return z.apply((val: number): number => (val < 0 ? this.params.leak * val : val));
  }


  public gradient(a: Vector, z: Vector): Vector {
    return z.apply((val: number): number => (val >= 0 ? 1 : this.params.leak));
  }


  public getParamSchema(): Joi.Schema {
    return Joi.object().keys(
      {
        leak: Joi.number().optional().default(0.0).description('Leak multiplier'),
      },
    );
  }
}
