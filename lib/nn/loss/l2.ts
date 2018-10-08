import Loss from './loss';
import Vector from '../../vector';

/**
 * Mean Squared Error / L2
 */
class L2 extends Loss
{
	calculate( yHat : Vector, y : Vector ) : number
	{
		// sum( ( yHat - y ) ^ 2 ) / y.size
		return yHat.sub( y ).pow( 2 ).sum() / y.getSize();
	}


	prime( yHat : Vector, y : Vector ) : Vector
	{
		return <Vector>yHat.sub( y );
	}
}


export default L2;
