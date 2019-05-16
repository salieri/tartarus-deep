import Joi from 'joi';
import { Initializer, InitializerParams } from './initializer';
import { NDArray } from '../../math';

export interface ConstantParams extends InitializerParams {
  value?: number;
}


export class Constant extends Initializer<ConstantParams> {
  protected readonly value: number;


  public constructor(params: ConstantParams) {
    super(params);

    this.value = params.value || 0;
  }


  public async initialize(data: NDArray): Promise<NDArray> {
    return data.set(this.value);
  }


  public getParamSchema(): Joi.Schema {
    return Joi.object().keys(
      {
        value: Joi.number().default(0.0).description('Constant value to initialize to'),
      },
    );
  }
}
