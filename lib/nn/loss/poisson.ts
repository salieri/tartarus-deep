import Loss from '.';
import Vector from '../../vector';

/**
 * Poisson
 */
class Poisson extends Loss
{
	calculate( yHat : Vector, y : Vector ) : number
	{
		// sum( yHat - y * log( yHat ) ) / y.size
		return yHat.sub( y.mul( yHat.log() ) ).mean();
	}
}


export default Poisson;
