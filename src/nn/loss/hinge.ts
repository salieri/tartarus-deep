import Joi from '@hapi/joi'; // Can't use JoiEx here -- circular dependency
import { Loss, LossParams } from './loss';
import { Vector } from '../../math';


export interface HingeParams extends LossParams {
  margin?: number;
}

export class Hinge extends Loss<HingeParams> {
  public calculate(yHat: Vector, y: Vector): number {
    return yHat.iterate(
      (values: number[]): number => {
        const yHatVal = values[0];
        const yVal = values[1];

        return Math.max(0, this.params.margin - yVal * yHatVal);
      },
      y,
    ).mean();
  }


  /* eslint-disable max-len */
  // tslint:disable max-line-length

  /**
   * https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/lossfunctions/impl/LossHinge.java
   */
  public gradient(yHat: Vector, y: Vector): Vector {
    return yHat.iterate(
      (values: number[]): number => {
        const yHatVal = values[0];
        const yVal = values[1];

        const yHatY = yHatVal * yVal;

        return (yHatY >= this.params.margin) ? 0 : (this.params.margin - yHatY);
      },
      y,
    );
  }


  public getParamSchema(): Joi.Schema {
    return Joi.object().keys(
      {
        margin: Joi.number().optional().default(1.0),
      },
    );
  }
}
