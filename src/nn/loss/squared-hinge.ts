import Joi from 'joi'; // Can't use JoiEx here -- circular dependency
import { Loss, LossDescriptor } from './loss';
import { NDArray, Vector } from '../../math';


export class SquaredHinge extends Loss {
  public calculate(yHat: Vector, y: Vector): number {
    return NDArray.iterate(
      (yHatVal: number, yVal: number): number => Math.max(0, this.params.margin - yVal * yHatVal) ** 2,
      yHat,
      y,
    ).mean();
  }


  public getDescriptor(): LossDescriptor {
    return {
      margin: Joi.number().optional().default(1.0),
    };
  }
}
