import _ from 'lodash';
import { GraphEntity } from './entity';
import { DeferredInputCollection } from '../symbols';


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

  private rawInputs?: DeferredInputCollection;

  private outputNodes: GraphNode[] = [];

  private inputNodes: GraphNode[] = [];

  // private connected: boolean = false;

  // private level: number = 0;


  public constructor(entity: GraphEntity) {
    this.entity = entity;
  }


  public getEntity(): GraphEntity {
    return this.entity;
  }


  public addOutputNode(node: GraphNode): void {
    this.outputNodes.push(node);
  }


  public addInputNode(node: GraphNode): void {
    this.inputNodes.push(node);
  }


  public removeOutput(node: GraphNode): void {
    _.remove(this.inputNodes, (input: GraphNode) => (node === input));
  }


  public removeInput(node: GraphNode): void {
    _.remove(this.outputNodes, (output: GraphNode) => (node === output));
  }


  public getInputNodes(): GraphNode[] {
    return this.inputNodes;
  }


  public getOutputNodes(): GraphNode[] {
    return this.outputNodes;
  }


  public getName(): string {
    return this.entity.getName();
  }


  public overrideRawInputs(inputs: DeferredInputCollection): void {
    this.rawInputs = inputs;
  }


  protected getRawInputs(): DeferredInputCollection {
    if (this.rawInputs) {
      return this.rawInputs;
    }

    const rawInputs = new DeferredInputCollection();

    _.each(
      this.inputNodes,
      (node: GraphNode) => rawInputs.merge(node.getEntity().getRawOutputs(), node.getName()),
    );

    this.rawInputs = rawInputs;

    return rawInputs;
  }


  public async compile(): Promise<void> {
    this.entity.setRawInputs(this.getRawInputs());

    await this.entity.compile();
  }
}

