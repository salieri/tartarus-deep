import _ from 'lodash';

import { GraphProcessorNode } from './node';
import { GraphNode } from '../node';

export type GraphNodeProcessorFunction = (node: GraphNode) => Promise<void>;
export type GraphNodeTestFunction = (node: GraphNode, direction: GraphProcessorDirection) => boolean;


export enum GraphProcessorDirection {
  Forward,
  Backward
}


export class GraphProcessor {
  private direction: GraphProcessorDirection;

  private nodes: GraphProcessorNode[];


  public constructor(nodes: GraphNode[], direction: GraphProcessorDirection) {
    this.nodes = _.map(nodes, (node: GraphNode) => (new GraphProcessorNode(node)));
    this.direction = direction;
  }


  protected async processReadyNodes(
    callback: GraphNodeProcessorFunction,
    canProcessTest: GraphNodeTestFunction = GraphProcessorNode.canProcess,
  ): Promise<void> {
    await Promise.all(
      _.map(
        _.filter(this.nodes, (node: GraphProcessorNode) => ((!node.isProcessed()) && (canProcessTest(node.getNode(), this.direction)))),
        async (node: GraphProcessorNode): Promise<void> => {
          node.setResult(await callback(node.getNode()));
          node.setProcessed(true);
        },
      ),
    );
  }


  protected countUnprocessedNodes(): number {
    return _.reduce(
      this.nodes,
      (accumulator: number, node: GraphProcessorNode): number => (accumulator + (node.isProcessed() ? 0 : 1)),
      0,
    );
  }


  protected unsetOutput(): void {
    _.each(this.nodes, (node: GraphProcessorNode) => node.unsetOutput(this.direction));
  }


  public async process(
    callback: GraphNodeProcessorFunction,
    canProcessTest: GraphNodeTestFunction = GraphProcessorNode.canProcess,
  ): Promise<void> {
    this.unsetOutput();

    let unprocessedCount = this.countUnprocessedNodes();

    while (unprocessedCount > 0) {
      // eslint-disable-next-line no-await-in-loop
      await this.processReadyNodes(callback, canProcessTest);

      const progressCount = this.countUnprocessedNodes();

      if ((progressCount >= unprocessedCount) && (unprocessedCount !== 0)) {
        throw new Error('Cannot resolve graph execution order');
      }

      unprocessedCount = progressCount;
    }
  }
}
