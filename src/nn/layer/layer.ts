import _ from 'lodash';

import {
  JoiEx,
  JoiExSchema,
  ContextLogger,
  Logger,
  MuteLogger,
} from '../../util';

import { Parameterized, Parameters } from '../../generic';


import { Session } from '../session';

import {
  CompilationStage,
  GraphEntity,
  GraphDataFeed,
  GraphRawFeed,
} from '../graph';


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
  public static readonly ERROR_TERM: string = 'error';

  public static readonly TRAINING_LABEL: string = 'train';

  private static layerCounter: number = 0;


  public readonly raw = new GraphRawFeed();

  public readonly data = new GraphDataFeed();


  protected readonly name: string;

  protected state: LayerState = LayerState.Created;

  protected session?: Session;

  protected logger: Logger = new MuteLogger();


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


  protected async compileInitialization(): Promise<void> { /* empty */ }

  protected async compileForwardPropagation(): Promise<void> { /* empty */ }

  protected async compileBackPropagation(): Promise<void> { /* empty */ }

  protected async compileFinalization(): Promise<void> { /* empty */ }


  protected async compileAsMember(stage: CompilationStage): Promise<void> {
    this.requireState(LayerState.Created);

    switch (stage) {
      case CompilationStage.ForwardPropagation:
        await this.compileForwardPropagation();
        break;

      case CompilationStage.BackPropagation:
        await this.compileBackPropagation();
        break;

      case CompilationStage.Initialize:
        await this.compileInitialization();
        break;

      case CompilationStage.Finalize:
        await this.compileFinalization();
        this.state = LayerState.Compiled;
        break;

      default:
        throw new Error(`Unknown compilation stage: ${stage}`);
    }
  }


  protected async compileAsMaster(): Promise<void> {
    // eslint-disable-next-line guard-for-in, no-restricted-syntax
    for (const stage in _.filter(CompilationStage, _.isNumber)) {
      // eslint-disable-next-line no-await-in-loop
      await this.compileAsMember(Number(stage));
    }
  }


  public async compile(stage?: CompilationStage): Promise<void> {
    this.requireState(LayerState.Created);

    await (_.isUndefined(stage) ? this.compileAsMaster() : this.compileAsMember(stage));
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


  public setLogger(parentLogger: Logger): void {
    this.logger = new ContextLogger(parentLogger, this.getName());
  }


  protected requireState(state: LayerState): void {
    if (this.state !== state) {
      throw new Error(`Unexpected state: ${LayerState[this.state]}`);
    }
  }
}

