import { Initializer } from './initializer';
import { NDArray } from '../../math';


export class One extends Initializer {
  public async initialize(data: NDArray): Promise<NDArray> {
    return data.set(1.0);
  }
}
