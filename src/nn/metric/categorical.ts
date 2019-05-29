import { Metric } from './metric';
import { Vector } from '../../math';

export class Categorical extends Metric {
  public calculate(yHat: Vector, y: Vector): number {
    return (yHat.argmax()[0] === y.argmax()[0]) ? 1 : 0;
  }
}


// K.cast(
//   K.equal(
//     K.argmax(y_true, axis=-1),
//     K.argmax(y_pred, axis=-1)
//   ),
//   K.floatx()
// )

