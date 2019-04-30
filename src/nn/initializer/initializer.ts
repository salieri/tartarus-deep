import { NDArray } from '../../math';
import Joi from 'joi';


export interface InitializerParams {
  [key: string]: any;
}

export interface InitializerDescriptor {
  [key: string]: any;
}


export abstract class Initializer {
  protected params: InitializerParams;

  constructor(params: InitializerParams = {}) {
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

