import Joi from '@hapi/joi';

import { Metric, MetricParams } from './metric';
import { Vector } from '../../math';


export interface TopKCategoricalParams extends MetricParams {
  k?: number;
}


export class TopKCategorical extends Metric<TopKCategoricalParams> {
  public calculate(yHat: Vector, y: Vector): number {
    return yHat.inTopK(y.argmax()[0], this.params.k) ? 1 : 0;
  }


  public getParamSchema(): Joi.Schema {
    return Joi.object().keys(
      {
        k: Joi.number().optional().default(5),
      },
    );
  }
}


// K.mean(
//   K.in_top_k(
//     y_pred,
//     K.argmax(y_true, axis=-1),
//     k
//   ),
//   axis=-1
// )

