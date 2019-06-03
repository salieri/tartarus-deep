import { GraphNode } from '../node';
import { DeferredInputCollection } from '../../symbols/deferred';
import { NodeConnector } from './connector';


export class NodeInputConnector extends NodeConnector {
  public getSourceNodesForNode(node: GraphNode): GraphNode[] {
    return node.getInputNodes();
  }


  public getNodeRawInputs(node: GraphNode): DeferredInputCollection {
    return node.getRawInputs();
  }


  public getSharedInputs(): DeferredInputCollection {
    return this.graph.getRawInputs();
  }


  public getMergeableSourcesForNode(sourceNode: GraphNode, targetNode: GraphNode): DeferredInputCollection {
    return sourceNode.getRawOutputs().filter([targetNode.getName(), DeferredInputCollection.DEFAULT_INPUT], true);
  }


  public getType(): string {
    return 'input';
  }
}
