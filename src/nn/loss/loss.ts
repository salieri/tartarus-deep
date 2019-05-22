import Joi from 'joi';
import { Vector } from '../../math';
import { Parameterized, Parameters } from '../../generic';

export type LossParams = Parameters;


/**
 * A loss function returns a scalar measuring
 * the performance of a model. Inputs are:
 * predicted label (yHat) and actual label (y)
 */
export abstract class Loss<T extends LossParams = LossParams> extends Parameterized<T> {
  public abstract calculate(yHat: Vector, y: Vector): number;


  public getParamSchema(): Joi.Schema {
    return Joi.object();
  }
}
