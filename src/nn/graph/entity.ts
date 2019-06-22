import { GraphNode } from './node';
import { Session } from '../session';
import { Logger } from '../../util';
import { GraphDataFeed } from './data-feed';
import { GraphRawFeed } from './raw-feed';
import { Optimizer } from '../optimizer';
import { Loss } from '../loss';

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

  data: GraphDataFeed;
  raw: GraphRawFeed;

  forward(): Promise<void>;
  backward(loss: Loss): Promise<void>;
  optimize(optimizer: Optimizer): Promise<void>;

  setSession(session: Session): void;
  setLogger(parentLogger: Logger): void;
}


export type EntityIdentifier = GraphNode|GraphEntity|string|number;

