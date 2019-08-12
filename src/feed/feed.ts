import { DeferredInputCollection } from '../nn';

// @link https://github.com/Microsoft/TypeScript/wiki/Breaking-Changes#extending-built-ins-like-error-array-and-map-may-no-longer-work
export class EndOfStreamException extends Error {
  public constructor(m: string) {
    super(m);

    Object.setPrototypeOf(this, EndOfStreamException.prototype);
  }
}


export interface Sample {
  name?: string;
  raw: DeferredInputCollection;
}


export interface Label {
  raw: DeferredInputCollection;
}


export interface InputFeedRecord {
  name    : string;
  offset  : number;
  count   : number;
  sample  : Sample;
  label?  : Label;
}


export abstract class InputFeed {
  /**
   * Prepare next set of sample data
   */
  public abstract async next(): Promise<InputFeedRecord>;


  /**
   * Count total number of sample sets
   */
  public abstract count(): number;


  /**
   * Return 0-index-based position of current sample set
   */
  public abstract offset(): number;


  /**
   * Set position to `offset`
   */
  public abstract async seek(offset: number): Promise<void>;


  /**
   * Has more records
   */
  public hasMore(): boolean {
    return this.offset() < this.count();
  }


  /**
   * Determine whether sample data contains labels
   */
  public abstract hasLabels(): boolean;
}
