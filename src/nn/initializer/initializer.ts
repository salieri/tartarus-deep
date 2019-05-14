import Joi from 'joi';
import { NDArray } from '../../math';
import { Layer } from '../layer';


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

  protected layer?: Layer;


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


  public attach(layer: Layer): void {
    if (this.layer) {
      throw new Error(`Initializer is already attached to layer '${layer.getLayerName()}'`);
    }

    this.layer = layer;
  }


  public abstract async initialize(data: NDArray): Promise<NDArray>;


  public getDescriptor(): InitializerDescriptor {
    return {};
  }
}

