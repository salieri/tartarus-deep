

export interface GraphEntity {
  compile(): Promise<void>;

  getName(): string;
}

