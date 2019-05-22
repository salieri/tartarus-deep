import { DeferredInputCollection } from '../symbols';
import { GraphNode } from './node';

/* eslint-disable-next-line @typescript-eslint/no-empty-interface */
export interface GraphEntityParams {}


export interface GraphEntity {
  compile(): Promise<void>;

  getName(): string;

  getRawOutputs(): DeferredInputCollection;
  setRawInputs(inputs: DeferredInputCollection): void;

  hasRawInputs(): boolean;
  hasRawOutputs(): boolean;
}


export type EntityIdentifier = GraphNode|GraphEntity|string|number;

