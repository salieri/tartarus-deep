import { Cost } from './cost';
import { NDArray } from '../../math';


export class Mean extends Cost {
  public calculate(lossScores: NDArray): number {
    return lossScores.mean();
  }
}
