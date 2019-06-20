import { DeferredCollection, DeferredCollectionWrapper } from '../symbols';
import { Vector } from '../../math';


export class GraphDataFeed {
  public readonly output = new DeferredCollection();

  public readonly input = new DeferredCollectionWrapper();

  public readonly backpropOutput = new DeferredCollection();

  public readonly backpropInput = new DeferredCollectionWrapper();

  public readonly fitter = new DeferredCollection();

  public readonly optimizer = new DeferredCollection();

  public readonly trainer = new DeferredCollectionWrapper();

  public activationDerivative?: Vector;


  public unsetIterationValues(): void {
    this.output.unsetValues();
    this.input.unsetValues();

    this.backpropOutput.unsetValues();
    this.backpropInput.unsetValues();

    this.trainer.unsetValues();
    this.fitter.unsetValues();
  }
}
