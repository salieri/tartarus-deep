import Loss from './';
import Vector from '../../vector';

/**
 * Mean Squared Error
 */
class MeanSquaredError extends Loss
{
	calculate( yHat : Vector, y : Vector ) : number
	{
		// sum( ( yHat - y ) ^ 2 ) / y.size
		return yHat.sub( y ).pow( 2 ).mean();
	}
}


export default MeanSquaredError;
