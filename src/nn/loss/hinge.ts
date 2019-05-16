import Joi from 'joi'; // Can't use JoiEx here -- circular dependency
import { Loss, LossParams } from './loss';
import { Vector, NDArray } from '../../math';


export interface HingeParams extends LossParams {
  margin?: number;
}


export class Hinge extends Loss<HingeParams> {
  protected readonly margin: number;


  public constructor(params: HingeParams = {}) {
    super(params);

    this.margin = this.params.margin || 1.0;
  }


  public calculate(yHat: Vector, y: Vector): number {
    return NDArray.iterate(
      (yHatVal: number, yVal: number): number => Math.max(0, this.margin - yVal * yHatVal),
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
