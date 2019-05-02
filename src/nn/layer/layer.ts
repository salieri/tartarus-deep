import { Promise } from 'bluebird';
import _ from 'lodash';
import { NDArray } from '../../math';
import { JoiEx } from '../../util';

import { NDSymbol, SymbolCollection } from '../symbol';


export interface LayerParams {
  [key: string]: any;
}

export interface LayerDescriptor {
  [key: string]: any;
}


export type DeferredValueType = NDArray;

export class DeferredValue {
  private value: DeferredValueType|null = null;

  private dimensions: number[]|null = null;


  public constructor(dimensions?: number[]|number) {
    if (typeof dimensions !== 'undefined') {
      this.declare(dimensions);
    }
  }


  public declare(dimensions: number[]|number): void {
    this.dimensions = _.castArray(dimensions);
  }


  public set(value: NDArray): void {
    this.mustBeDeclared();

    if (_.isEqual(this.dimensions, value.getDims())) {
      throw new Error('Value does not match expected dimensions');
    }

    this.value = value;
  }


  public size(): number {
    this.mustBeDeclared();

    return _.reduce(
      this.dimensions,
      (total, dimensionSize) => (total * dimensionSize),
      0,
    );
  }


  public getDims(): number[] {
    this.mustBeDeclared();

    return _.cloneDeep(this.dimensions as number[]);
  }


  public get(): DeferredValueType {
    this.mustBeDeclared();

    if (!this.value) {
      throw new Error('Value has not been set');
    }

    return this.value;
  }


  private mustBeDeclared(): void {
    if (this.dimensions === null) {
      throw new Error('Value has not been declared yet');
    }
  }
}


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
    this.collection[key].set(value);
  }
}


export abstract class Layer {
  public params = new LayerParams();

  public cache = new LayerCache();

  public input = new DeferredValue();

  public output = new DeferredValue();

  public optimizer = new DeferredCollection();

  public name: string;

  protected compiled: boolean = false;

  private static layerCounter: number = 0;



  public constructor(params: LayerParams = {}, name?: string) {
    this.params = this.validateParams(params);

    Layer.layerCounter += 1;

    this.name = name || `${this.constructor.name}#${Layer.layerCounter}`;
  }


  private validateParams(params: LayerParams): LayerParams {
    const result = JoiEx.validate(params, this.getDescriptor());

    if (result.error) {
      throw result.error;
    }

    return result.value;
  }


  public setParam(paramName: string, value: any): Layer {
    const result = JoiEx.validate(this.params[paramName], value);

    if (result.error) {
      throw result.error;
    }

    this.params[paramName] = result.value;

    return this;
  }


  public getDescriptor(): LayerDescriptor {
    return {};
  }


  public calculate(x: NDArray): NDArray {
    return x;
  }


  public forward(input: NDArray) {
  }


  public backward(output: NDArray) {
  }


  public getSymbol(name: string): any {
  }


  public setSymbol(name: string, symbol: Symbol) {

  }

  public hasSymbol(name: string) {

  }


  protected mustHaveSymbol(name: string): void {

  }


  public register(variableName: string, symbol: NDSymbol): void {
    this.symbols.add(this.getSymbolName(variableName), symbol);
  }


  public getSymbolName(variableName: string): string {
    return `${this.getLayerName()}-${variableName}`;
  }


  public getLayerName(): string {
    return this.name;
  }


  protected canModify(): void {
    if (this.compiled) {
      throw new Error('Layer cannot be modified after compilation');
    }
  }


  public compile(): void {
    this.canModify();

    this.compiled = true;
  }
}

