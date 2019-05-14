import _ from 'lodash';
import { GraphNode } from './node';
import { GraphEntity } from './entity';


export enum GraphState {
  Created,
  Compiling,
  Compiled,
  Initialized,
}


export class Graph {
  protected nodes: GraphNode[] = [];

  protected state: GraphState = GraphState.Created;


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


  /**
   * Add a entity to the graph
   * @param entity
   * @param [parentEntity] If specified, connects `entity` to the output of `parentEntity`
   */
  public add(entity: GraphEntity, parentEntity?: GraphEntity): GraphNode {
    this.canModify();

    if (this.exists(entity)) {
      throw new Error(`Entity '${entity.getName()}' already exists in this graph`);
    }

    if (parentEntity) {
      if (this.exists(parentEntity)) {
        throw new Error(`Parent entity '${entity.getName()}' does not exist in this graph`);
      }

      if (parentEntity === entity) {
        throw new Error('Parent entity cannot be same instance than the entity being added');
      }
    }

    const node = new GraphNode(entity);

    this.nodes.push(node);

    if (parentEntity) {
      this.link(parentEntity, entity);
    }

    return node;
  }


  /**
   * Link two neurons
   * @param outputEntity
   * @param inputEntity
   */
  public link(outputEntity: GraphEntity, inputEntity: GraphEntity): void {
    this.canModify();

    const outputNode  = this.find(outputEntity);
    const inputNode   = this.find(inputEntity);

    this.checkForCircularLinks(outputNode, 'backward', inputNode);
    this.checkForCircularLinks(outputNode, 'forward', inputNode);
    this.checkForCircularLinks(inputNode, 'backward', outputNode);
    this.checkForCircularLinks(inputNode, 'forward', outputNode);

    outputNode.addOutput(inputNode);
    inputNode.addInput(outputNode);
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
        throw new Error('Circular graph of nodes detected');
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
      (direction === 'forward') ? node.outputs : node.inputs,
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
  public exists(entity: GraphEntity): boolean {
    try {
      this.find(entity);

      return true;
    } catch (e) {
      if (e.match(/Unknown entity/)) {
        return false;
      }

      throw e;
    }
  }


  /**
   * Find entity in the graph
   * @param {GraphEntity|string|number} entity If `string`, matches against the name of the entity;
   * if `number` accesses graph entity pool by index;
   * if `GraphEntity` finds the node that matches the specific instance
   */
  public find(entity: GraphEntity | string | number): GraphNode {
    if (_.isNumber(entity) === true) {
      const nEntity = entity as number;

      if ((nEntity < 0) || (nEntity >= this.nodes.length)) {
        throw new Error(`Unknown entity index: ${nEntity}`);
      }

      return this.nodes[nEntity];
    }

    let node;

    if (_.isString(entity) === true) {
      node = _.find(this.nodes, (n: GraphNode) => (n.getEntity().getName() === entity as string));
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


  public async compile(): Promise<void> {
    if (this.state !== GraphState.Created) {
      throw new Error('Unexpected state');
    }

    this.state = GraphState.Compiling;

    await Promise.all(
      _.map(
        this.nodes,
        (node: GraphNode) => (node.getEntity().compile()),
      ),
    );

    this.state = GraphState.Compiled;
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

