import { DeferredValue, DeferredValueType } from './deferred-value';

export interface DeferredValueCollectionInf {
  [key: string]: DeferredValue;
}


export class DeferredCollection {
  private collection: DeferredValueCollectionInf = {};


  public declare(key: string, dimensions:number[]|number): void {
    if (key in this.collection) {
      throw new Error(`Key '${key}' has already been declared`);
    }

    this.collection[key] = new DeferredValue(dimensions);
  }


  public get(key: string): DeferredValue {
    if (!(key in this.collection)) {
      throw new Error(`Unknown key: '${key}'`);
    }

    return this.collection[key];
  }


  public getValue(key: string): DeferredValueType {
    if (!(key in this.collection)) {
      throw new Error(`Unknown key: '${key}'`);
    }

    return this.collection[key].get();
  }


  public setValue(key: string, value: DeferredValueType): void {
    if (!(key in this.collection)) {
      throw new Error(`Unknown key: '${key}'`);
    }

    this.collection[key].set(value);
  }
}
