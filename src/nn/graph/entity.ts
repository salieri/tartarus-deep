import { DeferredReadonlyCollection } from '../symbol';
import { GraphNode } from './node';

export interface GraphEntity {
  compile(): Promise<void>;

  getName(): string;

  getRawOutputs(): DeferredReadonlyCollection[];
  setRawInputs(inputs: DeferredReadonlyCollection[]): void;


  hasInputs(): boolean;
  hasOutputs(): boolean;
}


export type EntityIdentifier = GraphNode|GraphEntity|string;

