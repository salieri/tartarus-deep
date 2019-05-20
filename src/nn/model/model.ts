import Joi from 'joi';
import _ from 'lodash';
import { Graph, GraphEntity, GraphNode } from '../graph';
import { Session } from '../session';
import { Randomizer } from '../../math';
import { Parameterized } from '../../util';
import { DeferredReadonlyCollection, DeferredCollection } from '../symbol';

export enum ModelState {
  Created,
  Compiling,
  Compiled,
  Initialized,
}


export interface ModelParams {
  seed?: string;
}




export class Model
  extends Parameterized<ModelParams>
  implements GraphEntity {
  private static modelCounter: number = 0;

  protected state: ModelState = ModelState.Created;

  protected graph: Graph = new Graph();

  protected session: Session;

  protected readonly name: string;

  protected rawOutputs: DeferredCollection[] = [];

  protected rawInputs: DeferredReadonlyCollection[] = [];


  public constructor(params: ModelParams = {}, name?: string) {
    super(params);

    this.session  = new Session(this.params.seed);

    Model.modelCounter += 1;

    this.name = name || `${this.constructor.name}#${Model.modelCounter}`;
  }


  public getParamSchema(): Joi.Schema {
    return Joi.object().keys(
      {
        seed: Joi.string().optional().default('hello-world').min(2),
      },
    );
  }


  public getRandomizer(): Randomizer {
    return this.session.getRandomizer();
  }


  public getSession(): Session {
    return this.session;
  }


  public getName(): string {
    return this.name;
  }


  public add(entity: GraphEntity, parentEntities?: EntityIdentifier): Model {
    this.graph.add(entity, parentEntities);

    return this;
  }


  public hasInputs(): boolean {
    return !!_.find(this.rawInputs, (ri: DeferredReadonlyCollection) => (ri.getRequiredFields().length > 0));
  }


  public hasOutputs(): boolean {
    return !!_.find(this.rawOutputs, (ro: DeferredCollection) => (ro.getKeys().length > 0));
  }


  public setRawInputs(inputs: DeferredReadonlyCollection[]): void {
    this.rawInputs = inputs;
  }


  public getRawOutputs(): DeferredReadonlyCollection[] {
    return _.map(this.rawOutputs, (ro: DeferredCollection) => new DeferredReadonlyCollection(ro));
  }


  public push(entity: GraphEntity): Model {
    this.graph.push(entity);

    return this;
  }


  public input(dims: number[]|number, layerId = '__default'): Model {
    const finalDimensions = _.castArray(dims);

    this.knownInputs.declare(layerId, finalDimensions);

    return this;
  }


  public output(layerId: string|string[]): Model {
    _.each(
      _.castArray(layerId),
      (l: string) => this.knownOutputs.declare(l),
    );

    return this;
  }


  protected resolveInput(): void {

  }


  protected resolveOutput(): void {

  }


  public async compile(): Promise<void> {
    if (this.state !== ModelState.Created) {
      throw new Error('Model has already been compiled');
    }

    this.state = ModelState.Compiling;

    this.resolveInput();
    this.resolveOutput();

    await this.graph.compile(this.knownInputs);

    this.rawOutputs = this.graph.getOutputs();

    this.state = ModelState.Compiled;
  }


  public fit(): void {
  }


  public evaluate(): void {
  }


  public predict(): void {
  }
}

export default Model;
