import { Loss } from './loss';
import { Vector } from '../../vector';

/**
 * Mean Squared Error
 */
export class MeanSquaredError extends Loss
{
	calculate( yHat : Vector, y : Vector ) : number
	{
		// sum( ( yHat - y ) ^ 2 ) / y.size
		return yHat.sub( y ).pow( 2 ).mean();
	}
}

