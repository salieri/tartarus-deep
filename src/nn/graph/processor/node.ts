import { GraphNode } from '../node';
import { GraphProcessorDirection } from './processor';

/* eslint-disable-next-line */
export type GraphProcessorResult = any;


export class GraphProcessorNode {
  private node: GraphNode;

  private result?: GraphProcessorResult;

  private processed: boolean = false;

  public constructor(node: GraphNode) {
    this.node = node;
  }


  public setProcessed(processed: boolean): void {
    this.processed = processed;
  }


  public isProcessed(): boolean {
    return this.processed;
  }


  public static canProcess(node: GraphNode, direction: GraphProcessorDirection): boolean {
    switch (direction) {
      case GraphProcessorDirection.Forward:
        return node.getRawInputs().areAllSet();

      case GraphProcessorDirection.Backward:
        return node.getRawBackpropInputs().areAllSet();

      default:
        throw new Error('Unsupported direction');
    }
  }


  public unsetOutput(direction: GraphProcessorDirection): void {
    switch (direction) {
      case GraphProcessorDirection.Forward:
        this.node.unsetOutputValues();
        break;

      case GraphProcessorDirection.Backward:
        this.node.unsetBackpropOutputValues();
        break;

      default:
        throw new Error('Unsupported direction');
    }
  }


  public getNode(): GraphNode {
    return this.node;
  }


  public getResult(): GraphProcessorResult {
    return this.result;
  }


  public setResult(result: GraphProcessorResult): void {
    this.result = result;
  }
}

