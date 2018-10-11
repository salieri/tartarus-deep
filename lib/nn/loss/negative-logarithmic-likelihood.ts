import { Loss } from './loss';
import { Vector } from '../../vector';

/**
 * Negative Logarithmic Likelihood
 */
export class NegativeLogarithmicLikelihood extends Loss
{
	calculate( yHat : Vector, y : Vector ) : number
	{
		// -sum( log( yHat ) ) / y.size
		return -yHat.log().mean();
	}
}

