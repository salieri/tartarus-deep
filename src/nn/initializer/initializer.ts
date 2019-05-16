import Joi from 'joi';
import { NDArray } from '../../math';
import { Layer } from '../layer';
import { Parameterized, Parameters } from '../../util';


export type InitializerParams = Parameters;


export abstract class Initializer<T extends InitializerParams = InitializerParams> extends Parameterized<T> {
  protected layer?: Layer;

  public attach(layer: Layer): void {
    if (this.layer) {
      throw new Error(`Initializer is already attached to layer '${layer.getLayerName()}'`);
    }

    this.layer = layer;
  }


  public abstract async initialize(data: NDArray): Promise<NDArray>;


  public getParamSchema(): Joi.Schema {
    return Joi.object();
  }
}

