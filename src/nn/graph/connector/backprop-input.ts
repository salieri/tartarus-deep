import { GraphNode } from '../node';
import { DeferredInputCollection } from '../../symbols/deferred';
import { NodeConnector } from './connector';


export class NodeBackpropInputConnector extends NodeConnector {
  public listRelevantNodesForNode(node: GraphNode): GraphNode[] {
    return node.getOutputNodes();
  }


  public getNodeFeed(node: GraphNode): DeferredInputCollection {
    return node.getRawBackpropInputs();
  }


  public getSharedFeed(): DeferredInputCollection {
    return this.graph.getRawBackpropInputs();
  }


  public getMergeableSourcesForNode(node: GraphNode): DeferredInputCollection {
    return node.getRawBackpropOutputs();
  }


  public getType(): string {
    return 'backprop';
  }
}
