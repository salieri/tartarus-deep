import { Metric } from './metric';
import { Vector } from '../../math';

export class SparseCategorical extends Metric {
  public calculate(yHat: Vector, y: Vector): number {
    return (y.max() === yHat.argmax()[0]) ? 1 : 0;
  }
}

// K.cast(
//   K.equal(
//     K.max(y_true, axis=-1),
//     K.cast(
//       K.argmax(y_pred, axis=-1),
//       K.floatx()
//     )
//   ),
//   K.floatx()
// )

