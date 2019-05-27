import { Promise } from 'bluebird';
import { DeferredCollection, DeferredInputCollection, DeferredReadonlyCollection } from '../symbols';

import {
  JoiEx,
  JoiExSchema,
} from '../../util';

import { Parameterized, Parameters } from '../../generic';


import { Session } from '../session';
import { GraphEntity } from '../graph';


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
export abstract class Layer <TInput extends LayerParams = LayerParams, TCoerced extends TInput = TInput>
  extends Parameterized<TInput, TCoerced>
  implements GraphEntity {
  public readonly output = new DeferredCollection();

  public readonly optimizer = new DeferredCollection();

  protected readonly name: string;

  private static layerCounter: number = 0;

  protected state: LayerState = LayerState.Created;

  protected session?: Session;

  protected rawInputs: DeferredInputCollection = new DeferredInputCollection();


  public constructor(params: TInput = {} as any, name?: string) {
    super(params);

    Layer.layerCounter += 1;

    this.name = this.validateName(name || `${this.constructor.name}#${Layer.layerCounter}`);
  }


  protected validateName(name: string): string {
    if (name.match(/[. ]/)) {
      throw new Error('Layer names may not contain spaces or periods');
    }

    return name;
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


  public getName(): string {
    return this.name;
  }


  protected abstract async backwardExec(): Promise<void>;

  public async backward(): Promise<void> {
    this.requireState(LayerState.Initialized);

    await this.backwardExec();
  }


  protected abstract async forwardExec(): Promise<void>;

  public async forward(): Promise<void> {
    this.requireState(LayerState.Initialized);

    await this.forwardExec();
  }


  protected abstract async compileExec(): Promise<void>;

  public async compile(): Promise<void> {
    this.requireState(LayerState.Created);

    await this.compileExec();

    this.state = LayerState.Compiled;
  }


  protected abstract async initializeExec(): Promise<void>;

  public async initialize(): Promise<void> {
    this.requireState(LayerState.Compiled);

    await this.initializeExec();

    this.state = LayerState.Initialized;
  }


  public getSession(): Session {
    if (!this.session) {
      throw new Error(`Cannot resolve session: Layer '${this.getName()}' is not attached to a session`);
    }

    return this.session;
  }


  public setSession(session: Session): void {
    this.session = session;
  }


  public setRawInputs(inputs: DeferredInputCollection): void {
    this.rawInputs = inputs;
  }


  public getRawOutputs(): DeferredInputCollection {
    const out = new DeferredInputCollection();

    out.setDefault(this.output);

    return out;
  }


  public getRawInputs(): DeferredInputCollection {
    return this.rawInputs;
  }


  public getOptimizer(): DeferredReadonlyCollection {
    return new DeferredReadonlyCollection(this.optimizer);
  }


  protected requireState(state: LayerState): void {
    if (this.state !== state) {
      throw new Error(`Unexpected state: ${LayerState[this.state]}`);
    }
  }
}

