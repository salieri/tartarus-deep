import { Metric } from './metric';
import { Vector } from '../../math';

export class Binary extends Metric {
  public calculate(yHat: Vector, y: Vector): number {
    return yHat.round().equal(y).mean();
  }
}


// K.mean(
//   K.equal(
//     y_true,
//     K.round(y_pred)
//   ),
//   axis=-1
// )

