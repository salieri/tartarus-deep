import { Promise } from 'bluebird';
import { DeferredCollection, DeferredValue } from '../symbol';
import { JoiEx } from '../../util';

export enum LayerState {
  Created,
  Compiled,
  Initialized,
}


export interface LayerParams {
  [key: string]: any;
}


export interface LayerDescriptor {
  [key: string]: any;
}


/**
 * ```js
 * const l = new Layer();
 *
 * await l.compile();
 * await l.initialize();
 *
 * ...
 *
 * await l.forward();
 * await l.backward();
 *
 * ```
 */
export abstract class Layer {
  public readonly params: LayerParams = {};

  public readonly cache = new DeferredCollection();

  public readonly input = new DeferredValue();

  public readonly output = new DeferredValue();

  public readonly optimizer = new DeferredCollection();

  protected readonly name: string;

  private static layerCounter: number = 0;

  protected state: LayerState = LayerState.Created;


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


  protected abstract async backwardExec(): Promise<void>;

  public async backward(): Promise<void> {
    if (this.state !== LayerState.Initialized) {
      throw new Error(`Unexpected state: ${LayerState[this.state]}`);
    }

    await this.backwardExec();
  }


  protected abstract async forwardExec(): Promise<void>;

  public async forward(): Promise<void> {
    if (this.state !== LayerState.Initialized) {
      throw new Error(`Unexpected state: ${LayerState[this.state]}`);
    }

    await this.forwardExec();
  }


  protected abstract async compileExec(): Promise<void>;

  public async compile(): Promise<void> {
    if (this.state !== LayerState.Created) {
      throw new Error(`Unexpected state: ${LayerState[this.state]}`);
    }

    await this.compileExec();

    this.state = LayerState.Compiled;
  }


  protected abstract async initializeExec(): Promise<void>;

  public async initialize(): Promise<void> {
    if (this.state !== LayerState.Compiled) {
      throw new Error(`Unexpected state: ${LayerState[this.state]}`);
    }

    await this.initializeExec();

    this.state = LayerState.Initialized;
  }


  /*
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
  */


  public getLayerName(): string {
    return this.name;
  }
}

