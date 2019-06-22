import { Loss } from './loss';
import { Vector } from '../../math';

/**
 * Mean Absolute Error
 * @link https://stats.stackexchange.com/questions/312737/mean-absolute-error-mae-derivative
 */
export class MeanAbsoluteError extends Loss {
  public calculate(yHat: Vector, y: Vector): number {
    // sum( abs( yHat - y ) ) / y.size
    return yHat.sub(y).abs().mean();
  }


  public gradient(yHat: Vector, y: Vector): Vector {
    return yHat.iterate(
      (values: number[]): number => {
        const yHatVal = values[0];
        const yVal = values[1];

        return Math.sign(yHatVal - yVal);
      },
      y,
    );
  }
}

