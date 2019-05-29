import _ from 'lodash';
import { GraphNode } from './node';
import { GraphEntity, EntityIdentifier } from './entity';
import { DeferredInputCollection, DeferredReadonlyCollection } from '../symbols';
import { KeyNotFoundError } from '../../error';
import { GraphProcessor, GraphProcessorDirection } from './processor';


export enum GraphState {
  Created,
  Compiling,
  Compiled,
  Initialized,
}


export class Graph {
  protected nodes: GraphNode[] = [];

  protected state: GraphState = GraphState.Created;

  protected name: string;

  // protected rawOutputs: DeferredInputCollection = new DeferredInputCollection();

  protected rawInputs: DeferredInputCollection = new DeferredInputCollection();

  protected outputNodes: GraphNode[] = [];


  public constructor(name: string) {
    this.name = name;
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
  }


  public async compile(): Promise<void> {
    this.requireState(GraphState.Created);

    this.state = GraphState.Compiling;

    this.prevalidateGraph();
    this.resolveRawInputsForNodes();

    await Promise.all(
      _.map(
        this.nodes,
        (node: GraphNode) => (node.compile()), // no 'await' on purpose
      ),
    );

    // await this.verifyLinks();

    this.state = GraphState.Compiled;
  }


  protected resolveRawInputsForNodes(): void {
    let defaultInput: DeferredReadonlyCollection|undefined;
    let defaultSpent = false;

    try {
      defaultInput = this.rawInputs.getDefault();
    } catch (err) {
      if (!(err instanceof KeyNotFoundError)) {
        throw err;
      }
    }


    _.each(
      this.nodes,
      (node: GraphNode) => {
        const inputNodes = node.getInputNodes();
        const inputCount = inputNodes.length;
        const outputCount = node.getOutputNodes().length;

        if ((inputCount === 0) && (outputCount === 0) && (!_.find(this.outputNodes, (n: GraphNode) => (n === node)))) {
          throw new Error(`Node '${node.getName()}' is not connected with any other node or output`);
        }

        let entityInput;
        const nodeRawInputs = node.getRawInputs();

        // Feed matching input name from rawInputs to the node
        try {
          entityInput = this.rawInputs.get(node.getName());

          nodeRawInputs.setDefault(entityInput);
        } catch (err) {
          if (!(err instanceof KeyNotFoundError)) {
            throw err;
          }
        }

        // If no matching input was available in rawInputs and the node has no linked inputs, feed default input
        if ((inputCount === 0) && (!entityInput)) {
          if (!defaultInput) {
            throw new Error(`Could not resolve input for layer '${node.getName()}' in model ${this.getName()}`);
          }

          if (defaultSpent) {
            throw new Error(
              `Multiple inputs in model '${this.getName()}' require default inputs, including '${node.getName()}'`,
            );
          }

          nodeRawInputs.setDefault(defaultInput);

          defaultSpent = true;
        }

        // Feed linked inputs to the node
        _.each(
          inputNodes,
          (inputNode: GraphNode) => nodeRawInputs.merge(inputNode.getRawOutputs(), inputNode.getName()),
        );
      },
    );

    if ((defaultInput) && (!defaultSpent)) {
      throw new Error(`Default input was defined but not spent in model '${this.getName()}'`);
    }
  }


  public setRawInputs(inputs: DeferredInputCollection): void {
    this.rawInputs = inputs;
  }


  public getRawInputs(): DeferredInputCollection {
    return this.rawInputs;
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
      throw new Error(`Unexpected state: ${this.state}`);
    }
  }


  public assign(inputs: DeferredInputCollection): void {
    const expectedKeys = this.rawInputs.getKeys().sort();
    const inputKeys = inputs.getKeys().sort();
    const diff = _.difference(expectedKeys, inputKeys);

    if (diff.length > 0) {
      throw new Error(`Input is missing missing keys: ${diff}`);
    }

    this.rawInputs.assign(inputs);
  }


  public async forward(): Promise<void> {
    this.requireState(GraphState.Compiled);

    const processor = new GraphProcessor(this.nodes, GraphProcessorDirection.Forward);

    await processor.process(
      async (node: GraphNode) => (node.forward()),
    );
  }


  public async backward(): Promise<void> {
    this.requireState(GraphState.Compiled);

    const processor = new GraphProcessor(this.nodes, GraphProcessorDirection.Forward);

    await processor.process(
      async (node: GraphNode) => (node.backward()),
    );
  }


  public unsetOutputValues(): void {
    _.each(this.nodes, (node: GraphNode) => node.unsetOutputValues());
  }


  /**
   * This is super unoptimized way of compiling stuff, but intention is to
   * let the graph partially compile, which may enable other parts of the graph to compile,
   * and repeat that until the entire graph compiles.
   */
  // not necessary for compilation, actually, but might be the calculation method
  // public async compile(): Promise<void> {
  //   this.canModify();
  //
  //   this.state = GraphState.Compiling;
  //
  //   let uncompiledCount = 0;
  //   const compilationErrors = [];
  //
  //   do {
  //     const uncompiledLayers = _.map(
  //       _.filter(this.nodes, (node: LayerGraphNode) => (node.isCompiled())),
  //       (node: LayerGraphNode) => node.layer.compile(),
  //     );
  //
  //     try {
  //       /* eslint-disable no-await-in-loop */
  //       await Promise.all(uncompiledLayers);
  //     } catch (err) {
  //       if (err instanceof RecoverableCompilationError) {
  //         compilationErrors.push(err);
  //       } else {
  //         throw err;
  //       }
  //     }
  //
  //     uncompiledCount = this.countUnresolvedNodes();
  //
  //     if ((uncompiledCount > 0) && (uncompiledCount === uncompiledLayers.length)) {
  //       const e = new Error('Compilation failed');
  //
  //       e.compilationErrors = compilationErrors;
  //
  //       throw e;
  //     }
  //   } while (uncompiledCount > 0);
  //
  //
  //
  //   this.state = GraphState.Compiled;
  // }
}

