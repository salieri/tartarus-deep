import Joi from '@hapi/joi';
import { NDArray } from '../../math';
import { Layer } from '../layer';
import { Parameterized, Parameters } from '../../generic';


export type InitializerParams = Parameters;


export abstract class Initializer
  <TInput extends InitializerParams = InitializerParams, TCoerced extends TInput = TInput> extends Parameterized<TInput, TCoerced> {
  protected layer?: Layer;

  public attachLayer(layer: Layer): void {
    if (this.layer) {
      throw new Error(`Initializer is already attached to layer '${layer.getName()}'`);
    }

    this.layer = layer;
  }


  public abstract async initialize(data: NDArray): Promise<NDArray>;


  public getParamSchema(): Joi.Schema {
    return Joi.object();
  }
}

