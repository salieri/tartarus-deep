import Joi from 'joi';
import _ from 'lodash';

import {
  EntityIdentifier,
  Graph,
  GraphEntity,
  GraphNode,
} from '../graph';

import { Session } from '../session';
import { NDArray, Randomizer } from '../../math';
import { Parameterized } from '../../generic';
import { DeferredCollection, DeferredInputCollection } from '../symbols';

export enum ModelState {
  Created,
  Compiling,
  Compiled,
  Initialized,
}


export interface ModelParams {
  seed?: string;
}


export type RelaxedInputCollectionDefinition = number[]|number|NDArray|DeferredInputCollection|DeferredCollection;


export class Model
  extends Parameterized<ModelParams>
  implements GraphEntity {
  private static modelCounter: number = 0;

  protected state: ModelState = ModelState.Created;

  protected graph: Graph;

  protected session: Session;

  protected readonly name: string;


  public constructor(params: ModelParams = {}, name?: string) {
    super(params);

    this.session  = new Session(this.params.seed);

    Model.modelCounter += 1;

    this.name = this.validateName(name || `${this.constructor.name}#${Model.modelCounter}`);

    this.graph = new Graph(this.name);
  }


  protected validateName(name: string): string {
    if (name.match(/[. ]/)) {
      throw new Error('Model names may not contain spaces or periods');
    }

    return name;
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


  public add(entity: GraphEntity, parentEntities?: EntityIdentifier|EntityIdentifier[]): Model {
    this.graph.add(entity, parentEntities);

    return this;
  }


  public link(output: EntityIdentifier, input: EntityIdentifier): Model {
    this.graph.link(output, input);

    return this;
  }


  public setRawInputs(inputs: DeferredInputCollection): void {
    // this.rawInputs = inputs;
    this.graph.setRawInputs(inputs);
  }


  public getRawInputs(): DeferredInputCollection {
    return this.graph.getRawInputs();
  }


  public getRawOutputs(): DeferredInputCollection {
    return this.graph.getRawOutputs();
  }


  public getOutputNodes(): GraphNode[] {
    return this.graph.getOutputNodes();
  }


  public push(entity: GraphEntity): Model {
    const node = this.graph.push(entity);

    this.output(node);

    return this;
  }


  protected prepareInputCollection(definition: RelaxedInputCollectionDefinition): DeferredInputCollection {
    if (definition instanceof  DeferredInputCollection) {
      return definition;
    }

    if (definition instanceof DeferredCollection) {
      return new DeferredInputCollection(definition);
    }

    if (definition instanceof NDArray) {
      return new DeferredInputCollection(new DeferredCollection(definition));
    }

    const dims = _.castArray(definition);
    const collection = new DeferredCollection();

    collection.declareDefault(dims);

    return new DeferredInputCollection(collection);
  }


  public input(definition: RelaxedInputCollectionDefinition): Model {
    this.setRawInputs(this.prepareInputCollection(definition));

    return this;
  }


  public output(entities: EntityIdentifier|EntityIdentifier[]): Model {
    this.graph.setOutputNodes(entities);

    return this;
  }


  public getState(): ModelState {
    return this.state;
  }


  public async compile(): Promise<void> {
    if (this.state !== ModelState.Created) {
      throw new Error('Model has already been compiled');
    }

    this.state = ModelState.Compiling;

    await this.graph.compile();

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
