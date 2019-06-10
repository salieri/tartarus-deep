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
 * @link https://en.wikipedia.org/wiki/Activation_function
 */
export class ELU extends Activation<ELUParamsInput, ELUParamsCoerced> {
  public calculate(z: NDArray): NDArray {
    const leak = this.params.leak;

    return z.apply((val: number): number => (val > 0 ? val : leak * (Math.exp(val) - 1)));
  }


  public derivative(a: NDArray, z: NDArray): NDArray {
    const leak = this.params.leak;

    return NDArray.iterate(
      (arrayValues: number[]): number => {
        const aVal = arrayValues[0];
        const zVal = arrayValues[1];

        return (zVal > 0 ? 1 : aVal + leak);
      },
      a,
      z,
    );
  }


  public getParamSchema(): Joi.Schema {
    return Joi.object().keys(
      {
        leak: Joi.number().optional().default(0.0).description('Leak multiplier'),
      },
    );
  }
}
