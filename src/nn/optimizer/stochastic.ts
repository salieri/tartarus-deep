import Joi from 'joi';

import { Optimizer } from './optimizer';
import { Matrix, Vector } from '../../math';

export interface StochasticParams {
  rate?: number;
}


/**
 * https://towardsdatascience.com/how-does-back-propagation-in-artificial-neural-networks-work-c7cad873ea7
 * https://towardsdatascience.com/step-by-step-tutorial-on-linear-regression-with-stochastic-gradient-descent-1d35b088a843
 * https://adventuresinmachinelearning.com/stochastic-gradient-descent/
 */

export class Stochastic extends Optimizer<StochasticParams> {
  public optimize<NType extends Matrix|Vector>(weights: NType, dLossOverDWeights: NType): NType {
    // Wnew = WCurrent - a * (dLoss/dWeights)
    return weights.sub(dLossOverDWeights.mul(this.params.rate));
  }

  public getParamSchema(): Joi.Schema {
    return Joi.object().keys(
      {
        rate: Joi.number().optional().default(0.01),
      },
    );
  }
}

