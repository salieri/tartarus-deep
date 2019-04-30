import { Vector } from '../../math';

export interface LossParams {
  /* eslint-disable-next-line */
  [key: string]: any;
}

export interface LossDescriptor {
  /* eslint-disable-next-line */
  [key: string]: any;
}


/**
 * A loss function returns a scalar measuring
 * the performance of a model. Inputs are:
 * predicted label (yHat) and actual label (y)
 */
export abstract class Loss {
  protected params: LossParams;

  public constructor(params: LossParams = {}) {
    this.params = params;
  }

  public abstract calculate(yHat: Vector, y: Vector): number;


  public getDescriptor(): LossDescriptor {
    return {};
  }
}
