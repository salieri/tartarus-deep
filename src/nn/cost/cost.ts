import { NDArray } from '../../math';

export interface CostParams {
  [key: string]: any;
}


export interface CostDescriptor {
  [key: string]: any;
}


export abstract class Cost {
  protected params: CostParams;

  public constructor(params: CostParams) {
    this.params = params;
  }


  public abstract calculate(lossScores: NDArray): number;


  public getDescriptor(): CostDescriptor {
    return {};
  }
}

