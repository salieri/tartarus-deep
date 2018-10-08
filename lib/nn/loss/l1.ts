import Loss from './loss';
import Vector from '../../vector';

/**
 * Mean Absolute Error / L1
 */
class L1 extends Loss
{
	calculate( yHat : Vector, y : Vector ) : number
	{
		// sum( abs( yHat - y ) ) / size
		return yHat.sub( y ).abs().sum() / y.getSize();
	}
}


export default L1;
