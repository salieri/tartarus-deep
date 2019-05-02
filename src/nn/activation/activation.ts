import { NDArray } from '../../math';
import * as ActivationCtors from './index';


/**
 * Activation function `g` takes net input `z`
 * and outputs non-linear result `a` that determines
 * how active each layer in the input should be
 *
 * z = wx + b
 * a = g( z )
 */

/* eslint-disable-next-line */
type ActivationParamType = number|string|boolean;

/* eslint-disable-next-line */
type ActivationDescriptorType = any;


export interface ActivationParams {
  [key: string]: ActivationParamType;
}

export interface ActivationDescriptor {
  [key: string]: ActivationDescriptorType;
}


export abstract class Activation {
  protected params: ActivationParams;


  public constructor(params: ActivationParams = {}) {
    this.params = params;
  }


  public abstract calculate(z: NDArray): NDArray;


  public getDescriptor(): ActivationDescriptor {
    return {};
  }
}
