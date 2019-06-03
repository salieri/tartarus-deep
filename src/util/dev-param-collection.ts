import { KeyNotFoundError } from '../error';

export type DevParamValue = string|number|boolean|null|undefined;

export interface DevParamValueCollection {
  [key: string]: DevParamValue;
}

export class DevParamCollection {
  private params: DevParamValueCollection = {};


  public set(key: string, value: DevParamValue): void {
    this.params[key] = value;
  }


  public get(key: string): DevParamValue {
    if (!(key in this.params)) {
      throw new KeyNotFoundError(`Missing key '${key}'`, key);
    }

    return this.params[key];
  }
}
