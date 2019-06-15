import { DeferredInputCollection } from '../symbols';


export class GraphRawFeed {
  public inputs = new DeferredInputCollection();

  public outputs = new DeferredInputCollection();

  public backpropInputs = new DeferredInputCollection();

  public backpropOutputs = new DeferredInputCollection();

  public trainingLabels = new DeferredInputCollection();
}

