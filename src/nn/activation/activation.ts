import Joi from 'joi';
import { NDArray } from '../../math';
import { Parameterized, Parameters } from '../../util';


/**
 * Activation function `g` takes net input `z`
 * and outputs non-linear result `a` that determines
 * how active each layer in the input should be
 *
 * z = wx + b
 * a = g( z )
 */

export type ActivationParams = Parameters;


export abstract class Activation
  <TInput extends ActivationParams = ActivationParams, TCoerced extends TInput = TInput> extends Parameterized<TInput, TCoerced> {
  public abstract calculate(z: NDArray): NDArray;

  public getParamSchema(): Joi.Schema {
    return Joi.object();
  }
}
