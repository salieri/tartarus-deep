import Joi from 'joi';
import { NDArray } from '../../math';
import { Parameterized, Parameters } from '../../util';

export type CostParams = Parameters;


export abstract class Cost<T extends CostParams = CostParams> extends Parameterized<T> {
  public abstract calculate(lossScores: NDArray): number;


  public getParamSchema(): Joi.Schema {
    return Joi.object();
  }
}

