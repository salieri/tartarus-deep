import _ from 'lodash';
import { Layer } from '../layer';


export class LayerGraphFeed {
  public node: LayerGraphNode;

  public label: string;

  public constructor(node: LayerGraphNode, label: string) {
    this.node = node;
    this.label = label;
  }
}


export class LayerGraphNode {
  private layer: Layer;

  private inputs: LayerGraphFeed[] = [];

  private outputs: LayerGraphFeed[] = [];

  private connected: boolean = false;

  private level: number = 0;


  public constructor(layer: Layer) {
    this.layer = layer;
  }


  public addOutput(node: LayerGraphNode): void {
    this.outputs.push(node);
  }


  public addInput(node: LayerGraphNode): void {
    this.inputs.push(node);
  }


  public removeOutput(node: LayerGraphNode): void {
    _.remove(this.inputs, (input: LayerGraphNode) => (node === input));
  }


  public removeInput(node: LayerGraphNode): void {
    _.remove(this.outputs, (output: LayerGraphNode) => (node === output));
  }
}


