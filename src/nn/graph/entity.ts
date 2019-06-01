import { DeferredInputCollection } from '../symbols';
import { GraphNode } from './node';
import { Session } from '../session';
import { Logger } from '../../util';

/* eslint-disable-next-line @typescript-eslint/no-empty-interface */
export interface GraphEntityParams {}

export enum CompilationStage {
  Initialize,
  ForwardPropagation,
  BackPropagation,
  Finalize
}


export interface GraphEntity {
  compile(state: CompilationStage): Promise<void>;
  initialize(): Promise<void>;

  getName(): string;

  getRawOutputs(): DeferredInputCollection;
  getRawInputs(): DeferredInputCollection;
  setRawInputs(inputs: DeferredInputCollection): void;

  getRawBackpropOutputs(): DeferredInputCollection;
  getRawBackpropInputs(): DeferredInputCollection;
  setRawBackpropInputs(inputs: DeferredInputCollection): void;

  forward(): Promise<void>;
  backward(): Promise<void>;

  unsetOutputValues(): void;
  unsetBackpropOutputValues(): void;

  setSession(session: Session): void;
  setLogger(parentLogger: Logger): void;
}


export type EntityIdentifier = GraphNode|GraphEntity|string|number;

