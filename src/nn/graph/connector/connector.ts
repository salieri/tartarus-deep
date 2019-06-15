import _ from 'lodash';

import { GraphNode } from '../node';
import { DeferredInputCollection, DeferredCollectionWrapper } from '../../symbols/deferred';
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

  public abstract getSourceNodesForNode(node: GraphNode): GraphNode[];

  public abstract getNodeRawInputs(node: GraphNode): DeferredInputCollection;

  public abstract getSharedInputs(): DeferredInputCollection|null;

  public abstract getMergeableSourcesForNode(sourceNode: GraphNode, targetNode: GraphNode): DeferredInputCollection;

  public abstract getType(): string;


  public getDefault(): DeferredCollectionWrapper|null {
    try {
      const sharedInputs = this.getSharedInputs();

      if (!sharedInputs) {
        return null;
      }

      return sharedInputs.getDefault();
    } catch (err) {
      if (!(err instanceof KeyNotFoundError)) {
        throw err;
      }

      return null;
    }
  }


  public connect(): GraphNode[] {
    let defaultOutputSpent = false;
    const defaultRawOutput = this.getDefault();
    const detectedExternalInputNodes: GraphNode[] = [];
    const sharedInputs = this.getSharedInputs();

    this.logger.debug('connector.connect', () => ({ mode: this.getType() }));

    _.each(
      this.graph.getAllNodes(),
      (curNode: GraphNode) => {
        this.logger.debug('connector.connect.process', () => ({ node: curNode.getName() }));

        const sourceNodes = this.getSourceNodesForNode(curNode);
        const sourceNodeCount = sourceNodes.length;

        this.logger.debug(
          'connector.connect.sources',
          () => ({ node: curNode.getName(), relevant: _.map(sourceNodes, (n: GraphNode) => n.getName()) }),
        );

        let rawOutputs;
        const curNodeRawInputs = this.getNodeRawInputs(curNode);

        // Feed matching input name from rawInputs to the node
        try {
          if (sharedInputs) {
            rawOutputs = sharedInputs.get(curNode.getName());

            curNodeRawInputs.setDefault(rawOutputs);

            detectedExternalInputNodes.push(curNode);

            this.logger.debug('connector.connect.node.source.fromShared', () => ({ node: curNode.getName() }));
          }
        } catch (err) {
          if (!(err instanceof KeyNotFoundError)) {
            throw err;
          }
        }

        // If no matching input was available in rawInputs and the node has no linked inputs, feed default input
        if ((sourceNodeCount === 0) && (!rawOutputs) && (sharedInputs)) {
          if (!defaultRawOutput) {
            throw new Error(`Could not resolve ${this.getType()} entity for `
             + `layer '${curNode.getName()}' in model ${this.graph.getName()}`);
          }

          if (defaultOutputSpent) {
            throw new Error(
              `Multiple ${this.getType()} entities in model '${this.graph.getName()}' expect `
              + `default inputs, including '${curNode.getName()}'`,
            );
          }

          curNodeRawInputs.setDefault(defaultRawOutput);

          detectedExternalInputNodes.push(curNode);

          defaultOutputSpent = true;

          this.logger.debug('connector.connect.node.source.fromDefault', () => ({ node: curNode.getName() }));
        }

        // Feed linked sources to the node
        _.each(
          sourceNodes,
          (sourceNode: GraphNode) => {
            try {
              curNodeRawInputs.merge(
                this.getMergeableSourcesForNode(sourceNode, curNode),
                sourceNode.getName(),
                false,
                (key: string) => ((key === curNode.getName()) ? sourceNode.getName() : key),
              );
            } catch (err) {
              throw err;
            }
          },
        );

        this.logger.debug('connector.connect.node.feeds', () => ({ node: curNode.getName(), keys: curNodeRawInputs.getKeys() }));
      },
    );

    if ((defaultRawOutput) && (!defaultOutputSpent)) {
      throw new Error(`Default ${this.getType()} entity was defined but not spent in model '${this.graph.getName()}'`);
    }

    this.logger.debug('connector.connect.nodes', () => ({ nodes: _.map(detectedExternalInputNodes, (n: GraphNode) => n.getName()) }));

    return _.uniq(detectedExternalInputNodes);
  }
}
