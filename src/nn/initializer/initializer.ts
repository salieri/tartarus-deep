import Joi from 'joi';
import { NDArray } from '../../math';


export interface InitializerParams {
  /* eslint-disable-next-line */
  [key: string]: any;
}

export interface InitializerDescriptor {
  /* eslint-disable-next-line */
  [key: string]: any;
}


export abstract class Initializer {
  protected params: InitializerParams;

  public constructor(params: InitializerParams = {}) {
    this.params = this.getValidatedParams(params);
  }


  protected getValidatedParams(params: InitializerParams): InitializerParams {
    const result = Joi.validate(params, this.getDescriptor());

    if (result.error) {
      throw result.error;
    }

    return result.value;
  }


  public abstract initialize(data: NDArray): NDArray;


  public getDescriptor(): InitializerDescriptor {
    return {};
  }
}

