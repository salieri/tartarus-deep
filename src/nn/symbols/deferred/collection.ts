import _ from 'lodash';
import { DeferredValue } from './value';

import {
  Matrix,
  NDArray,
  NDArrayCollection,
  Vector,
} from '../../../math';

import { KeyNotFoundError } from '../../../error';


export interface DeferredValueCollectionInf {
  [key: string]: DeferredValue;
}


export class DeferredCollection {
  private static idCounter:number = 0;

  private collection: DeferredValueCollectionInf = {};

  private defaultKey: string = 'default';

  private id: number;

  public constructor(defaultValue?: NDArray|NDArrayCollection) {
    if (defaultValue) {
      this.coerceData(defaultValue);
    }

    DeferredCollection.idCounter += 1;

    this.id = DeferredCollection.idCounter;
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


  public getValue<NDType extends NDArray = NDArray>(key: string, type?: { new (val: NDArray|Matrix|Vector): NDType }): NDType {
    this.require(key);

    return this.collection[key].get(type);
  }


  public getKeys(): string[] {
    return _.keys(this.collection);
  }


  public setValue <NDType>(key: string, value: NDArray, type?: { new (val: NDArray|Matrix|Vector): NDType }): void {
    this.require(key);

    this.collection[key].set(value, type);
  }


  public setDefaultKey(key: string): void {
    this.require(key);

    this.defaultKey = key;
  }


  public setDefaultValue<NDType>(value: NDArray, type?: { new (val: NDArray|Matrix|Vector): NDType }): void {
    this.setValue(this.getDefaultKey(), value, type);
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


  public getDefaultValue<NDType extends NDArray = NDArray>(type?: { new (val: NDArray|Matrix|Vector): NDType }): NDType {
    return this.getDefault().get(type);
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


  public areAllDeclared(): boolean {
    return (this.getKeys().length > 0);
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


  public clone(): DeferredCollection {
    const copy = new DeferredCollection();

    copy.collection = _.mapValues(
      this.collection,
      (value: DeferredValue) => value.clone(),
    );

    copy.setDefaultKey(this.getDefaultKey());

    return copy;
  }


  public eachValue(cb: <T extends NDArray>(nd: T, fieldKey: string) => T|undefined|void): void {
    _.each(
      this.collection,
      (value: DeferredValue, fieldKey: string) => {
        const curValue = value.get();

        const result = cb(curValue, fieldKey);

        if (result) {
          value.set(result);
        }
      },
    );
  }
}
