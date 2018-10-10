import Loss from '.';
import Vector from '../../vector';

/**
 * Mean Absolute Error
 */
class MeanAbsoluteError extends Loss
{
	calculate( yHat : Vector, y : Vector ) : number
	{
		// sum( abs( yHat - y ) ) / y.size
		return yHat.sub( y ).abs().mean();
	}
}


export default MeanAbsoluteError;
