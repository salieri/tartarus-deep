import _ from 'lodash';

import { GraphNode } from '../node';
import { DeferredInputCollection, DeferredReadonlyCollection } from '../../symbols/deferred';
import { Graph } from '../graph';
import { KeyNotFoundError } from '../../../error';
import { ContextLogger, Logger } from '../../../util';

export abstract class NodeConnector {
  protected graph: Graph;

  protected logger: Logger;


  public constructor(graph: Graph) {
    this.graph = graph;

    this.logger = new ContextLogger(graph.getLogger(), 'connector');
  }

  public abstract listRelevantNodesForNode(node: GraphNode): GraphNode[];

  public abstract getNodeFeed(node: GraphNode): DeferredInputCollection;

  public abstract getSharedFeed(): DeferredInputCollection;

  public abstract getMergeableSourcesForNode(node: GraphNode): DeferredInputCollection;

  public abstract getType(): string;


  public getDefault(): DeferredReadonlyCollection|null {
    try {
      return this.getSharedFeed().getDefault();
    } catch (err) {
      if (!(err instanceof KeyNotFoundError)) {
        throw err;
      }

      return null;
    }
  }


  public connect(): GraphNode[] {
    let defaultSpent = false;
    const defaultRaw = this.getDefault();
    const fedNodes: GraphNode[] = [];

    this.logger.debug('connector.connect', () => ({ mode: this.getType() }));

    _.each(
      this.graph.getAllNodes(),
      (node: GraphNode) => {
        this.logger.debug('connector.connect.process', () => ({ node: node.getName() }));

        const relevantNodes = this.listRelevantNodesForNode(node);
        const relevantNodeCount = relevantNodes.length;

        this.logger.debug(
          'connector.connect.relevant',
          () => ({ node: node.getName(), relevant: _.map(relevantNodes, (rn: GraphNode) => rn.getName()) }),
        );

        let entityRaw;
        const nodeRaw = this.getNodeFeed(node);

        // Feed matching input name from rawInputs to the node
        try {
          entityRaw = this.getSharedFeed().get(node.getName());

          nodeRaw.setDefault(entityRaw);

          fedNodes.push(node);

          this.logger.debug('connector.connect.node.feed.fromShared', () => ({ node: node.getName() }));
        } catch (err) {
          if (!(err instanceof KeyNotFoundError)) {
            throw err;
          }
        }

        // If no matching input was available in rawInputs and the node has no linked inputs, feed default input
        if ((relevantNodeCount === 0) && (!entityRaw)) {
          if (!defaultRaw) {
            throw new Error(`Could not resolve ${this.getType()} entity for `
             + `layer '${node.getName()}' in model ${this.graph.getName()}`);
          }

          if (defaultSpent) {
            throw new Error(
              `Multiple ${this.getType()} entities in model '${this.graph.getName()}' expect `
              + `default inputs, including '${node.getName()}'`,
            );
          }

          nodeRaw.setDefault(defaultRaw);

          fedNodes.push(node);

          defaultSpent = true;

          this.logger.debug('connector.connect.node.feed.default', () => ({ node: node.getName() }));
        }

        // Feed linked sources to the node
        _.each(
          relevantNodes,
          (sourceNode: GraphNode) => nodeRaw.merge(this.getMergeableSourcesForNode(sourceNode), sourceNode.getName()),
        );


        this.logger.debug('connector.connect.node.feeds', () => ({ node: node.getName(), keys: nodeRaw.getKeys() }));
      },
    );

    if ((defaultRaw) && (!defaultSpent)) {
      throw new Error(`Default ${this.getType()} entity was defined but not spent in model '${this.graph.getName()}'`);
    }

    this.logger.debug('connector.connect.nodes', () => ({ nodes: _.map(fedNodes, (n: GraphNode) => n.getName()) }));

    return fedNodes;
  }
}
