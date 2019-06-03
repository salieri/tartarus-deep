import { GraphNode } from '../node';
import { DeferredInputCollection } from '../../symbols/deferred';
import { NodeConnector } from './connector';


export class NodeBackpropInputConnector extends NodeConnector {
  public getSourceNodesForNode(node: GraphNode): GraphNode[] {
    return node.getOutputNodes();
  }


  public getNodeRawInputs(node: GraphNode): DeferredInputCollection {
    return node.getRawBackpropInputs();
  }


  public getSharedInputs(): DeferredInputCollection {
    return this.graph.getRawBackpropInputs();
  }


  public getMergeableSourcesForNode(sourceNode: GraphNode, targetNode: GraphNode): DeferredInputCollection {
    return sourceNode.getRawBackpropOutputs().filter([targetNode.getName(), DeferredInputCollection.DEFAULT_INPUT], true);
  }


  public getType(): string {
    return 'backprop';
  }
}
