import { GraphNode } from './node';
import { Session } from '../session';
import { Logger } from '../../util';
import { GraphDataFeed } from './data-feed';
import { GraphRawFeed } from './raw-feed';

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
  backward(): Promise<void>;

  setSession(session: Session): void;
  setLogger(parentLogger: Logger): void;
}


export type EntityIdentifier = GraphNode|GraphEntity|string|number;

