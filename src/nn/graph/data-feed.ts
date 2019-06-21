import { DeferredCollection, DeferredCollectionWrapper } from '../symbols';


export class GraphDataFeed {
  /**
   * Values produced by forward propagation; wiped between iterations
   */
  public readonly output = new DeferredCollection();


  /**
   * Values required by forward propagation; wiped between iterations
   */
  public readonly input = new DeferredCollectionWrapper();


  /**
   * Values produced by backpropagation; wiped between iterations
   */
  public readonly backpropOutput = new DeferredCollection();


  /**
   * Values required by backpropagation; wiped between iterations
   */
  public readonly backpropInput = new DeferredCollectionWrapper();


  /**
   * Values that should be collected for mini-batch processing; wiped between iterations
   */
  public readonly fitter = new DeferredCollection();


  /**
   * Training labels; wiped between iterations
   */
  public readonly trainer = new DeferredCollectionWrapper();


  /**
   * Values that should be stored between sessions (or needed for restoration)
   */
  public readonly optimizer = new DeferredCollection();


  public unsetIterationValues(): void {
    this.output.unsetValues();
    this.input.unsetValues();

    this.backpropOutput.unsetValues();
    this.backpropInput.unsetValues();

    this.trainer.unsetValues();
    this.fitter.unsetValues();
  }
}
