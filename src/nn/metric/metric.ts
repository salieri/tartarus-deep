import Joi from '@hapi/joi';
import { Vector } from '../../math';
import { Parameterized, Parameters } from '../../generic';

export type MetricParams = Parameters;


export abstract class Metric<T extends MetricParams = MetricParams> extends Parameterized<T> {
  public abstract calculate(yHat: Vector, y: Vector): number;


  public getParamSchema(): Joi.Schema {
    return Joi.object();
  }
}
