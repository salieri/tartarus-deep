import _ from 'lodash';
import { GraphNode } from './node';
import { CompilationStage, EntityIdentifier, GraphEntity } from './entity';
import { DeferredCollection, DeferredInputCollection } from '../symbols';
import { GraphProcessor, GraphProcessorDirection } from './processor';
import { NodeBackpropInputConnector, NodeInputConnector } from './connector';
import { Layer } from '../layer';
import { Session } from '../session';
import { ContextLogger, Logger, MuteLogger } from '../../util';


export enum GraphState {
  Created,
  Compiled,
  Initialized,
}


export class Graph {
  protected nodes: GraphNode[] = [];

  protected state: GraphState = GraphState.Created;

  protected name: string;

  // protected rawOutputs: DeferredInputCollection = new DeferredInputCollection();

  protected rawInputs: DeferredInputCollection = new DeferredInputCollection();

  protected rawBackpropInputs: DeferredInputCollection = new DeferredInputCollection();

  protected outputNodes: GraphNode[] = [];

  protected inputNodes: GraphNode[] = [];

  protected logger: Logger = new MuteLogger();

  protected session: Session;


  public constructor(name: string, session: Session) {
    this.name = name;
    this.session = session;
  }


  /**
   * Attach entity to the last entry in the current graph.
   * Essentially creates a sequential graph.
   * @param entity
   */
  public push(entity: GraphEntity): GraphNode {
    let parentEntity;

    if (this.nodes.length > 0) {
      parentEntity = this.nodes[this.nodes.length - 1].getEntity();
    }

    return this.add(entity, parentEntity);
  }


  protected resolveEntities(entities?: EntityIdentifier|EntityIdentifier[]): GraphNode[] {
    if (!entities) {
      return [];
    }

    return _.map(
      _.castArray(entities),
      (entity: EntityIdentifier): GraphNode => {
        if (entity instanceof GraphNode) {
          return entity as GraphNode;
        }

        if (_.isString(entity)) {
          return this.find(entity as string);
        }

        return this.find(entity as GraphEntity);
      },
    );
  }


  /**
   * Add a entity to the graph
   * @param entity
   * @param [parentEntities] If specified, connects `entity` to the output of `parentEntity`
   */
  public add(entity: GraphEntity, parentEntities?: EntityIdentifier|EntityIdentifier[]): GraphNode {
    this.canModify();

    if (this.exists(entity)) {
      throw new Error(`Instance '${entity.getName()}' already exists in this graph`);
    }

    if (this.exists(entity.getName())) {
      throw new Error(`Duplicate entity name: '${entity.getName()}' already exists in this graph`);
    }

    const resolvedParents = this.resolveEntities(parentEntities);

    _.each(
      resolvedParents,
      (node: GraphNode) => {
        if (node.getEntity() === entity) {
          throw new Error('Parent entity cannot be same instance than the entity being added');
        }
      },
    );


    const newNode = new GraphNode(entity);

    this.nodes.push(newNode);

    _.each(resolvedParents, (parentNode: GraphNode) => (this.link(parentNode, newNode)));

    entity.setSession(this.session);
    entity.setLogger(this.logger);

    return newNode;
  }


  /**
   * Link two neurons
   * @param output
   * @param input
   */
  public link(output: EntityIdentifier, input: EntityIdentifier): void {
    this.canModify();

    const outputNode  = this.find(output);
    const inputNode   = this.find(input);

    this.checkForCircularLinks(outputNode, 'backward', inputNode);
    this.checkForCircularLinks(inputNode, 'forward', outputNode);

    // this.checkForCircularLinks(outputNode, 'forward', inputNode);
    // this.checkForCircularLinks(inputNode, 'backward', outputNode);

    outputNode.addOutputNode(inputNode);
    inputNode.addInputNode(outputNode);
  }


  /**
   * Test the graph does not become circular
   * @param node
   * @param direction
   * @param targetNode
   */
  private checkForCircularLinks(node: GraphNode, direction: string, targetNode: GraphNode): void {
    const linkTestFn = (linkNode: GraphNode): void => {
      if (linkNode === targetNode) {
        throw new Error(`Circular graph of nodes detected while attempting to link '${node.getName()}' to '${targetNode.getName()}'`);
      }
    };

    this.traverse(node, direction, linkTestFn);
  }


  /**
   * Traverse the graph starting from `node`
   * @param node
   * @param direction Direction to traverse (`'backward'` or `'forward'`)
   * @param {Function(GraphNode node)} callback Callback to be called for every node traversed
   */
  public traverse(node: GraphNode, direction: string, callback: Function): void {
    _.each(
      (direction === 'forward') ? node.getOutputNodes() : node.getInputNodes(),
      (linkNode: GraphNode): void => {
        callback(linkNode);
        this.traverse(linkNode, direction, callback);
      },
    );
  }


  /**
   * Unlink two entities
   * @param outputEntity
   * @param inputEntity
   */
  public unlink(outputEntity: GraphEntity, inputEntity: GraphEntity): void {
    this.canModify();

    const outputNode  = this.find(outputEntity);
    const inputNode   = this.find(inputEntity);

    outputNode.removeOutput(inputNode);
    inputNode.removeInput(outputNode);
  }


  /**
   * Remove entity from the graph
   * @param entity
   */
  public remove(entity: GraphEntity): void {
    this.canModify();

    const node = this.find(entity);

    _.each(
      this.nodes,
      (e: GraphNode) => {
        e.removeInput(node);
        e.removeOutput(node);
      },
    );

    _.remove(this.nodes, (e: GraphNode) => (e === node));
  }


  /**
   * Check if a entity exists in the graph
   * @param entity
   */
  public exists(entity: EntityIdentifier): boolean {
    try {
      this.find(entity);

      return true;
    } catch (e) {
      if ((e.message) && (e.message.match(/Unknown entity/))) {
        return false;
      }

      throw e;
    }
  }


  /**
   * Find entity in the graph
   * @param {EntityIdentifier} entity If `string`, matches against the name of the entity;
   * if `number` accesses graph entity pool by index;
   * if `GraphEntity` finds the node that matches the specific instance
   */
  public find(entity: EntityIdentifier): GraphNode {
    if (_.isNumber(entity) === true) {
      const nEntity = entity as unknown as number;

      if ((nEntity < 0) || (nEntity >= this.nodes.length)) {
        throw new Error(`Unknown entity index: ${nEntity}`);
      }

      return this.nodes[nEntity];
    }

    let node;

    if (_.isString(entity) === true) {
      node = _.find(this.nodes, (n: GraphNode) => (n.getName() === (entity as string)));
    } else if (entity instanceof GraphNode) {
      node = _.find(this.nodes, (n: GraphNode) => (n === entity));
    } else {
      node = _.find(this.nodes, (n: GraphNode) => (n.getEntity() === entity));
    }

    if (_.isUndefined(node)) {
      throw new Error(`Unknown entity: ${entity}`);
    }

    return node;
  }


  protected canModify(): void {
    if (this.state !== GraphState.Created) {
      throw new Error('Graph cannot be modified after compilation');
    }
  }


  protected prevalidateGraph(): void {
    if (this.outputNodes.length === 0) {
      throw new Error(`Model '${this.getName()}' has no defined outputs`);
    }

    this.checkForUnconnectedNodes();
  }


  public async compile(stage: CompilationStage): Promise<void> {
    this.requireState(GraphState.Created);

    switch (stage) {
      case CompilationStage.Initialize:
        this.prevalidateGraph();
        break;

      case CompilationStage.ForwardPropagation:
        this.resolveRawInputsForNodes();
        break;

      case CompilationStage.BackPropagation:
        this.resolveRawBackpropInputsForNodes();
        break;

      case CompilationStage.Finalize:
        break;

      default:
        throw new Error(`Unknown compilation stage: ${stage}`);
    }


    const processor = new GraphProcessor(
      this.nodes,
      (stage !== CompilationStage.BackPropagation) ? GraphProcessorDirection.Forward : GraphProcessorDirection.Backward,
    );

    await processor.process(
      async (node: GraphNode) => node.compile(stage),
      (node: GraphNode, direction: GraphProcessorDirection) => {
        switch (direction) {
          case GraphProcessorDirection.Forward:
            return node.getRawInputs().areAllDeclared();

          case GraphProcessorDirection.Backward:
            return node.getRawBackpropInputs().areAllDeclared();

          default:
            throw new Error('Unsupported direction');
        }
      },
    );


    if (stage === CompilationStage.Finalize) {
      this.state = GraphState.Compiled;
    }
  }


  protected resolveRawInputsForNodes(): void {
    const connector = new NodeInputConnector(this);

    this.inputNodes = connector.connect();
  }


  protected determineBackpropInputs(): DeferredInputCollection {
    const rawOutputs = this.getRawOutputs();
    const backpropInputs = new DeferredInputCollection();

    _.each(
      rawOutputs.getKeys(),
      (key: string) => {
        const output = rawOutputs.get(key).getDefault();
        const coll = new DeferredCollection();

        coll.declare(Layer.DERIVATIVE, output.getDims());
        coll.declare(Layer.LOSS, 1);

        backpropInputs.set(key, coll);
      },
    );

    return backpropInputs;
  }


  protected resolveRawBackpropInputsForNodes(): void {
    this.rawBackpropInputs = this.determineBackpropInputs();

    const connector = new NodeBackpropInputConnector(this);

    connector.connect();
  }


  protected checkForUnconnectedNodes(): void {
    _.each(
      this.nodes,
      (node: GraphNode) => {
        const inputCount = node.getInputNodes().length;
        const outputCount = node.getOutputNodes().length;

        if ((inputCount === 0) && (outputCount === 0) && (!_.find(this.outputNodes, (n: GraphNode) => (n === node)))) {
          throw new Error(`Node '${node.getName()}' is not connected with any other node or output`);
        }
      },
    );
  }


  public getAllNodes(): GraphNode[] {
    return this.nodes;
  }


  public setRawInputs(inputs: DeferredInputCollection): void {
    this.rawInputs = inputs;
  }


  public getRawInputs(): DeferredInputCollection {
    return this.rawInputs;
  }


  public setRawBackpropInputs(inputs: DeferredInputCollection): void {
    this.rawBackpropInputs = inputs;
  }


  public getRawBackpropInputs(): DeferredInputCollection {
    return this.rawBackpropInputs;
  }


  public getRawBackpropOutputs(): DeferredInputCollection {
    if (this.inputNodes.length === 0) {
      throw new Error('No input nodes');
    }

    if (this.inputNodes.length === 1) {
      const collection = new DeferredInputCollection();

      // don't re-map default output
      collection.merge(this.inputNodes[0].getRawBackpropOutputs());

      return collection;
    }

    const out = new DeferredInputCollection();

    _.each(
      this.inputNodes,
      (inputNode: GraphNode) => (out.merge(inputNode.getRawBackpropOutputs(), inputNode.getName())),
    );

    return out;
  }


  public getRawOutputs(): DeferredInputCollection {
    if (this.outputNodes.length === 0) {
      throw new Error('No output nodes');
    }

    if (this.outputNodes.length === 1) {
      const collection = new DeferredInputCollection();

      // don't re-map default output
      collection.merge(this.outputNodes[0].getRawOutputs());

      return collection;
    }

    const out = new DeferredInputCollection();

    _.each(
      this.outputNodes,
      (outputNode: GraphNode) => (out.merge(outputNode.getRawOutputs(), outputNode.getName())),
    );

    return out;
  }


  public getOutputNodes(): GraphNode[] {
    return this.outputNodes;
  }


  public setOutputNodes(entities: EntityIdentifier|EntityIdentifier[]): void {
    this.outputNodes = _.map(
      _.castArray(entities),
      (entity: EntityIdentifier) => (this.find(entity)),
    );
  }


  public getName(): string {
    return this.name;
  }


  protected requireState(state: GraphState): void {
    if (this.state !== state) {
      throw new Error(`Unexpected state: ${GraphState[this.state]}`);
    }
  }


  public assignInput(inputs: DeferredInputCollection): void {
    const expectedKeys = this.rawInputs.getKeys().sort();
    const inputKeys = inputs.getKeys().sort();
    const diff = _.difference(expectedKeys, inputKeys);

    if (diff.length > 0) {
      throw new Error(`Input is missing missing keys: ${diff}`);
    }

    this.rawInputs.assign(inputs);
  }


  public assignBackpropInput(inputs: DeferredInputCollection): void {
    const expectedKeys = this.rawBackpropInputs.getKeys().sort();
    const inputKeys = inputs.getKeys().sort();
    const diff = _.difference(expectedKeys, inputKeys);

    if (diff.length > 0) {
      throw new Error(`Backprop input is missing missing keys: ${diff}`);
    }

    this.rawBackpropInputs.assign(inputs);
  }


  public async initialize(): Promise<void> {
    this.requireState(GraphState.Compiled);

    await Promise.all(_.map(this.nodes, (n: GraphNode) => n.initialize()));

    this.state = GraphState.Initialized;
  }


  public async forward(): Promise<void> {
    this.requireState(GraphState.Initialized);

    const processor = new GraphProcessor(this.nodes, GraphProcessorDirection.Forward);

    await processor.process(
      async (node: GraphNode) => (node.forward()),
    );
  }


  public async backward(): Promise<void> {
    this.requireState(GraphState.Initialized);

    const processor = new GraphProcessor(this.nodes, GraphProcessorDirection.Backward);

    await processor.process(
      async (node: GraphNode) => (node.backward()),
    );
  }


  public unsetOutputValues(): void {
    _.each(this.nodes, (node: GraphNode) => node.unsetOutputValues());
  }


  public unsetInputValues(): void {
    _.each(this.nodes, (node: GraphNode) => node.unsetInputValues());
  }


  public unsetBackpropOutputValues(): void {
    _.each(this.nodes, (node: GraphNode) => node.unsetBackpropOutputValues());
  }


  public unsetBackpropInputValues(): void {
    _.each(this.nodes, (node: GraphNode) => node.unsetBackpropInputValues());
  }


  public setSession(session: Session): void {
    this.session = session;

    _.each(this.nodes, (node: GraphNode) => node.getEntity().setSession(session));
  }


  public setLogger(parentLogger: Logger): void {
    this.logger = new ContextLogger(parentLogger, 'graph');

    _.each(this.nodes, (node: GraphNode) => node.getEntity().setLogger(this.logger));
  }


  public getLogger(): Logger {
    return this.logger;
  }
}

