import { GraphNode } from '../node';
import { DeferredInputCollection } from '../../symbols/deferred';
import { NodeConnector } from './connector';


export class NodeInputConnector extends NodeConnector {
  public listRelevantNodesForNode(node: GraphNode): GraphNode[] {
    return node.getInputNodes();
  }


  public getNodeFeed(node: GraphNode): DeferredInputCollection {
    return node.getRawInputs();
  }


  public getSharedFeed(): DeferredInputCollection {
    return this.graph.getRawInputs();
  }


  public getMergeableSourcesForNode(node: GraphNode): DeferredInputCollection {
    return node.getRawOutputs();
  }


  public getType(): string {
    return 'input';
  }
}
