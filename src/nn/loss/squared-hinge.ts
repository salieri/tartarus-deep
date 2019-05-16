import Joi from 'joi'; // Can't use JoiEx here -- circular dependency
import { Loss, LossParams } from './loss';
import { NDArray, Vector } from '../../math';

export interface SquaredHingeParams extends LossParams {
  margin?: number;
}


export class SquaredHinge extends Loss<SquaredHingeParams> {
  protected readonly margin: number;


  public constructor(params: SquaredHingeParams = {}) {
    super(params);

    this.margin = params.margin || 1.0;
  }


  public calculate(yHat: Vector, y: Vector): number {
    return NDArray.iterate(
      (yHatVal: number, yVal: number): number => Math.max(0, this.margin - yVal * yHatVal) ** 2,
      yHat,
      y,
    ).mean();
  }


  public getParamSchema(): Joi.Schema {
    return Joi.object().keys(
      {
        margin: Joi.number().optional().default(1.0),
      },
    );
  }
}
