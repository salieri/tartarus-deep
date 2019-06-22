import { DeferredInputCollection } from '../nn';

export class EndOfStreamException extends Error {}


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


export abstract class DeferredInputFeed {
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
