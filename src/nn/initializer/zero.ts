import { Initializer } from './initializer';
import { NDArray } from '../../math';


export class Zero extends Initializer {

  public initialize(data: NDArray): NDArray {
    return data.set(0.0);
  }

}
