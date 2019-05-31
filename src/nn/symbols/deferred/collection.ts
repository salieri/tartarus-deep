import _ from 'lodash';
import { DeferredValue, DeferredValueType } from './value';
import { NDArray, NDArrayCollection } from '../../../math';
import { KeyNotFoundError } from '../../../error';


export interface DeferredValueCollectionInf {
  [key: string]: DeferredValue;
}


export class DeferredCollection {
  private collection: DeferredValueCollectionInf = {};

  private defaultKey: string = 'default';

  public constructor(defaultValue?: NDArray|NDArrayCollection) {
    if (defaultValue) {
      this.coerceData(defaultValue);
    }
  }


  protected coerceData(value: NDArray|NDArrayCollection): void {
    if (value instanceof NDArray) {
      this.declare(this.defaultKey, value.getDims());
      this.setDefaultValue(value);
      return;
    }

    _.each(
      value,
      (v: NDArray, k: string): void => {
        this.declare(k, v.getDims());
        this.setValue(k, v);
      },
    );
  }


  public declare(key: string, dimensions:number[]|number): void {
    if (key in this.collection) {
      throw new Error(`Key '${key}' has already been declared`);
    }

    this.collection[key] = new DeferredValue(dimensions);
  }


  public declareDefault(dimensions: number[]|number): void {
    this.declare(this.getDefaultKey(), dimensions);
  }


  public get(key: string): DeferredValue {
    this.require(key);

    return this.collection[key];
  }


  public getValue(key: string): DeferredValueType {
    this.require(key);

    return this.collection[key].get();
  }


  public getKeys(): string[] {
    return _.keys(this.collection);
  }


  public setValue(key: string, value: DeferredValueType): void {
    this.require(key);

    this.collection[key].set(value);
  }


  public setDefaultKey(key: string): void {
    this.require(key);

    this.defaultKey = key;
  }


  public setDefaultValue(value: DeferredValueType): void {
    this.setValue(this.getDefaultKey(), value);
  }


  public getDefaultKey(): string {
    return this.defaultKey;
  }


  public getDefault(): DeferredValue {
    if (!this.defaultKey) {
      throw new Error('Default key has not been set');
    }

    return this.get(this.defaultKey);
  }


  public getDefaultValue(): DeferredValueType {
    return this.getDefault().get();
  }


  public require(key: string): void {
    if (!(key in this.collection)) {
      throw new KeyNotFoundError(`Unknown key: '${key}'`, key);
    }
  }


  public requireDefault(): void {
    this.require(this.defaultKey);
  }


  public areAllSet(): boolean {
    return _.every(
      this.collection,
      (value: DeferredValue) => value.isSet(),
    );
  }


  public unsetValues(): void {
    _.each(this.collection, (value: DeferredValue) => value.unset());
  }


  public assign(inbound: DeferredCollection): void {
    const inboundDefaultKey = inbound.getDefaultKey();

    _.each(
      inbound.getKeys(),
      (key: string) => {
        const finalKey = (key === inboundDefaultKey) ? this.getDefaultKey() : key;

        if (!this.has(finalKey)) {
          throw new Error(`Cannot assign undeclared key '${key}' => '${finalKey}'`);
        }

        this.setValue(finalKey, inbound.get(key).get());
      },
    );
  }


  public has(key: string): boolean {
    return (key in this.collection);
  }
}
