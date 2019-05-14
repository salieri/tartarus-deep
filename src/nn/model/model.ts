import Joi from 'joi';
import { Graph, GraphNode } from '../graph';
import { Layer } from '../layer';
import { Session } from '../session';
import { Randomizer } from '../../math';


export interface ModelParams {
  seed?: string;
}

export interface ModelDescriptor {
  [key: string]: any;
}


export class Model {
  protected graph: Graph = new Graph();

  protected params: ModelParams;

  protected session: Session;


  public constructor(params: ModelParams = {}) {
    this.params   = this.getValidatedParams(params);
    this.session  = new Session(this.params.seed);
  }


  protected getValidatedParams(params: ModelParams): ModelParams {
    const result = Joi.validate(params, this.getDescriptor());

    if (result.error) {
      throw result.error;
    }

    return result.value;
  }


  public getDescriptor(): ModelDescriptor {
    return {
      seed: Joi.string().optional().default('hello-world').min(2),
    };
  }


  public getRandomizer(): Randomizer {
    return this.session.getRandomizer();
  }


  public add(layer: Layer, parentLayer?: Layer) {
    return this.graph.add(layer, parentLayer);
  }


  public push(layer: Layer) {
    return this.graph.push(layer);
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
