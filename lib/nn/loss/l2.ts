import Loss from '.';
import Vector from '../../vector';

/**
 * L2
 */
class L2 extends Loss
{
	calculate( yHat : Vector, y : Vector ) : number
	{
		// sum( ( yHat - y ) ^ 2 )
		return yHat.sub( y ).pow( 2 ).sum();
	}
}


export default L2;
