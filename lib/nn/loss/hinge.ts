import Loss from './loss';
import Vector from '../../vector';


class Hinge extends Loss
{
	calculate( yHat : Vector, y : Vector ) : number
	{
		//return 1 - ( yHat.matmul( y ) )
	}
}


export default Hinge;
