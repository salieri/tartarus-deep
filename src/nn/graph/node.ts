import _ from 'lodash';
import { GraphEntity } from './entity';


export class GraphFeed {
  public node: GraphNode;

  public label: string;

  public constructor(node: GraphNode, label: string) {
    this.node = node;
    this.label = label;
  }
}


export class GraphNode {
  private readonly entity: GraphEntity;

  private inputs: GraphFeed[] = [];

  private outputs: GraphFeed[] = [];

  // private connected: boolean = false;

  // private level: number = 0;


  public constructor(entity: GraphEntity) {
    this.entity = entity;
  }


  public getEntity(): GraphEntity {
    return this.entity;
  }



  public addOutput(node: GraphNode): void {
    this.outputs.push(node);
  }


  public addInput(node: GraphNode): void {
    this.inputs.push(node);
  }


  public removeOutput(node: GraphNode): void {
    _.remove(this.inputs, (input: GraphNode) => (node === input));
  }


  public removeInput(node: GraphNode): void {
    _.remove(this.outputs, (output: GraphNode) => (node === output));
  }
}


