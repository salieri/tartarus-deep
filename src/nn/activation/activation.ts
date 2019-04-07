import { NDArray } from '../../math';


/**
 * Activation function `g` takes net input `z`
 * and outputs non-linear result `a` that determines
 * how active each layer in the input should be
 *
 * z = wx + b
 * a = g( z )
 */

export interface ActivationParams {
  [key: string]: any;
}

export interface ActivationDescriptor {
  [key: string]: any;
}


export abstract class Activation {
  protected params: ActivationParams;


  constructor(params: ActivationParams = {}) {
    this.params = params;
  }


  public abstract calculate(z: NDArray): NDArray;


  public getDescriptor(): ActivationDescriptor {
    return {};
  }
}
