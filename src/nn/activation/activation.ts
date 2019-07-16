import Joi from 'joi';
import { Vector } from '../../math';
import { Parameterized, Parameters } from '../../generic';


/**
 * Activation function `g` takes linear input `z`
 * and outputs non-linear result `a`
 *
 * z = wx + b
 * a = g( z )
 */

export type ActivationParams = Parameters;


export abstract class Activation
  <TInput extends ActivationParams = ActivationParams, TCoerced extends TInput = TInput> extends Parameterized<TInput, TCoerced> {
  public abstract calculate(z: Vector): Vector;

  /**
   * @param a Activated output
   * @param z Linear output
   * @param y? Expected label
   */
  public abstract gradient(a: Vector, z: Vector, y?: Vector): Vector;

  public getParamSchema(): Joi.Schema {
    return Joi.object();
  }
}
