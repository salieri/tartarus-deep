import Loss from '.';
import Vector from '../../vector';

/**
 * Negative Logarithmic Likelihood
 */
class NegativeLogarithmicLikelihood extends Loss
{
	calculate( yHat : Vector, y : Vector ) : number
	{
		// -sum( log( yHat ) ) / y.size
		return -yHat.log().mean();
	}
}


export default NegativeLogarithmicLikelihood;
