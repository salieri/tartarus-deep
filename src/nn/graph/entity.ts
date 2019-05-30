import { DeferredInputCollection } from '../symbols';
import { GraphNode } from './node';
import { Session } from '../session';

/* eslint-disable-next-line @typescript-eslint/no-empty-interface */
export interface GraphEntityParams {}


export interface GraphEntity {
  compile(): Promise<void>;
  initialize(): Promise<void>;

  getName(): string;

  getRawOutputs(): DeferredInputCollection;
  getRawInputs(): DeferredInputCollection;

  setRawInputs(inputs: DeferredInputCollection): void;

  forward(): Promise<void>;
  backward(): Promise<void>;

  unsetOutputValues(): void;

  setSession(session: Session): void;
}


export type EntityIdentifier = GraphNode|GraphEntity|string|number;

