import { Loss, LossDescriptor } from './loss';
import { Vector, NDArray } from '../../math';
import Joi from 'joi'; // Can't use JoiEx here -- circular dependency


export class Hinge extends Loss {
  public calculate(yHat: Vector, y: Vector): number {
    return NDArray.iterate(
      (yHatVal: number, yVal: number): number => {
        return Math.max(0, this.params.margin - yVal * yHatVal);
      },
      yHat,
      y
    ).mean();
  }


  public getDescriptor(): LossDescriptor {
    return {
      margin: Joi.number().optional().default(1.0)
    };
  }
}
