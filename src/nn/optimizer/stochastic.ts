import Joi from 'joi';

import { Optimizer } from './optimizer';
import { Matrix, Vector } from '../../math';

export interface StochasticParams {
  rate?: number;
}

export class Stochastic extends Optimizer<StochasticParams> {
  public optimize<NType extends Matrix|Vector>(weights: NType, weightError: NType): NType {
    // W - (dW*rate)
    return weights.sub(weightError.mul(this.params.rate)) as NType;
  }

  public getParamSchema(): Joi.Schema {
    return Joi.object().keys(
      {
        rate: Joi.number().optional().default(0.01),
      },
    );
  }
}

