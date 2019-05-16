import Joi from 'joi'; // Can't use JoiEx here -- circular dependency
import { Loss, LossParams } from './loss';
import { Vector } from '../../math';


export interface HuberParams extends LossParams {
  delta?: number;
}


export class Huber extends Loss<HuberParams> {
  public calculate(yHat: Vector, y: Vector): number {
    const delta = this.params.delta;

    return yHat.apply(
      (yHatVal: number, pos: number[]): number => {
        const yVal = y.getAt(pos);

        if (yVal - yHatVal < delta) {
          return 0.5 * ((yVal - yHatVal) ** 2);
        }

        return delta * (yVal - yHatVal) - 0.5 * delta;
      },
    ).mean();
  }


  public getParamSchema(): Joi.Schema {
    return Joi.object().keys(
      {
        delta: Joi.number().optional().default(1.0),
      },
    );
  }
}

