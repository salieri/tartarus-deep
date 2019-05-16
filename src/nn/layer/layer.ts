import { Promise } from 'bluebird';
import { DeferredCollection, DeferredValue } from '../symbol';

import {
  JoiEx,
  JoiExSchema,
  Parameterized,
  Parameters,
} from '../../util';

import { Session } from '../session';


export enum LayerState {
  Created,
  Compiled,
  Initialized,
}


export type LayerParams = Parameters;


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
export abstract class Layer
  <TInput extends LayerParams = LayerParams, TCoerced extends TInput = TInput> extends Parameterized<TInput, TCoerced> {
  public readonly cache = new DeferredCollection();

  public readonly input = new DeferredValue();

  public readonly output = new DeferredValue();

  public readonly optimizer = new DeferredCollection();

  protected readonly name: string;

  private static layerCounter: number = 0;

  protected state: LayerState = LayerState.Created;

  protected model: Model;


  public constructor(params: TInput = {} as any, name?: string) {
    super(params);

    Layer.layerCounter += 1;

    this.name = name || `${this.constructor.name}#${Layer.layerCounter}`;
  }


  // public setParam(paramName: string, value: any): Layer {
  //   const result = JoiEx.validate(this.params[paramName], value);
  //
  //   if (result.error) {
  //     throw result.error;
  //   }
  //
  //   this.params[paramName] = result.value;
  //
  //   return this;
  // }


  public getParamSchema(): JoiExSchema {
    return JoiEx.object();
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


  public getSession(): Session {
    if (!this.model) {
      throw new Error(`Cannot resolve session: Layer ${this.name} is not attached to a model`);
    }

    return this.model.getSession();
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

