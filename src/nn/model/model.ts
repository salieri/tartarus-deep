import Joi from 'joi';
import { Graph, GraphEntity, GraphNode } from '../graph';
import { Session } from '../session';
import { Randomizer } from '../../math';
import { Parameterized } from '../../util';

export enum ModelState {
  Created,
  Compiling,
  Compiled,
  Initialized,
}


export interface ModelParams {
  seed?: string;
}


export class Model extends Parameterized<ModelParams> implements GraphEntity {
  private static modelCounter: number = 0;

  protected state: ModelState = ModelState.Created;

  protected graph: Graph = new Graph();

  protected session: Session;

  protected readonly name: string;


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


  public getName(): string {
    return this.name;
  }


  public add(entity: GraphEntity, parentEntity?: GraphEntity): Model {
    this.graph.add(entity, parentEntity);

    return this;
  }


  public push(entity: GraphEntity): GraphNode {
    return this.graph.push(entity);
  }


  public async compile(): Promise<void> {
    if (this.state !== ModelState.Created) {
      throw new Error('Model has already been compiled');
    }

    this.state = ModelState.Compiling;

    await this.graph.compile();

    this.state = ModelState.Compiled;
  }


  public fit() {
  }


  public evaluate() {
  }


  public predict() {
  }
}

export default Model;
