import Joi from 'joi'; // Can't use JoiEx here -- circular dependency
import { Loss, LossParams } from './loss';
import { NDArray, Vector } from '../../math';

export interface SquaredHingeParams extends LossParams {
  margin?: number;
}


export class SquaredHinge extends Loss<SquaredHingeParams> {
  public calculate(yHat: Vector, y: Vector): number {
    return NDArray.iterate(
      (values: number[]): number => {
        const yHatVal = values[0];
        const yVal = values[1];

        return Math.max(0, this.params.margin - yVal * yHatVal) ** 2;
      },
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
