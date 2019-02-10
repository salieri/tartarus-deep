import { Initializer, InitializerDescriptor } from './initializer';
import { NDArray } from '../../math';
import Joi from 'joi';


export class Constant extends Initializer {

  public initialize(data: NDArray): NDArray {
    return data.set(this.params.value);
  }


  public getDescriptor(): InitializerDescriptor {
    return {
      value: Joi.number().default(0.0).description('Constant value to initialize to')
    };
  }
}
