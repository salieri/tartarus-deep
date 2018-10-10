import Loss from '.';
import Vector from '../../vector';

/**
 * Mean Squared Logarithmic Error
 */
class MeanSquaredLogarithmicError extends Loss
{
	calculate( yHat : Vector, y : Vector ) : number
	{
		// sum( ( log( y + 1 ) - log( yHat + 1 ) ) ^ 2 ) / y.size

		const yHatPlusOneLog	= yHat.add( 1.0 ).log(),
			yPlusOneLog			= y.add( 1.0 ).log();

		return yPlusOneLog.sub( yHatPlusOneLog ).pow( 2 ).mean();
	}
}


export default MeanSquaredLogarithmicError;
