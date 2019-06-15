import _ from 'lodash';
import { CompilationStage, GraphEntity } from './entity';
import { DeferredInputCollection } from '../symbols';
import { DevParamCollection } from '../../util';


export class GraphNode {
  private readonly entity: GraphEntity;

  private rawInputs: DeferredInputCollection = new DeferredInputCollection();

  private rawBackpropInputs: DeferredInputCollection = new DeferredInputCollection();

  private outputNodes: GraphNode[] = [];

  private inputNodes: GraphNode[] = [];

  public devParams: DevParamCollection = new DevParamCollection();


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


  public getRawOutputs(): DeferredInputCollection {
    return this.entity.raw.outputs;
  }


  public getRawInputs(): DeferredInputCollection {
    return this.rawInputs;
  }


  public getRawBackpropOutputs(): DeferredInputCollection {
    return this.entity.raw.backpropOutputs;
  }


  public getRawBackpropInputs(): DeferredInputCollection {
    return this.rawBackpropInputs;
  }


  public getRawTrainingLabels(): DeferredInputCollection {
    return this.entity.raw.trainingLabels;
  }


  public unsetIterationValues(): void {
    this.entity.data.unsetIterationValues();
  }


  public assignTrainingLabels(labels: DeferredInputCollection): void {
    this.entity.raw.trainingLabels.assign(labels);
  }


  public async compile(stage: CompilationStage): Promise<void> {
    switch (stage) {
      case CompilationStage.Initialize:
        break;

      case CompilationStage.ForwardPropagation:
        this.entity.raw.inputs = this.rawInputs;
        break;

      case CompilationStage.BackPropagation:
        this.entity.raw.backpropInputs = this.rawBackpropInputs;
        break;

      case CompilationStage.Finalize:
        break;

      default:
        throw new Error(`Unknown compilation stage: ${stage}`);
    }

    await this.entity.compile(stage);
  }


  public async forward(): Promise<void> {
    await this.getEntity().forward();
  }


  public async backward(): Promise<void> {
    await this.getEntity().backward();
  }


  public async initialize(): Promise<void> {
    await this.getEntity().initialize();
  }
}
