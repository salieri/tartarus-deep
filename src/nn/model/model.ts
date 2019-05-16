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


export class Model extends Parameterized<ModelParams> {
  protected state: ModelState = ModelState.Created;

  protected graph: Graph = new Graph();

  protected session: Session;


  public constructor(params: ModelParams = {}) {
    super(params);

    this.session  = new Session(this.params.seed);
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


  public add(entity: GraphEntity, parentEntity?: GraphEntity): GraphNode {
    return this.graph.add(entity, parentEntity);
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
