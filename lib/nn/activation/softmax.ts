import Activation from './';
import Vector from '../../vector';

/**
 * Softmax
 */
class Softmax extends Activation
{
	calculate( yHat : Vector, y : Vector ) : Vector
	{
		return <Vector>yHat.exp().div( yHat.exp().sum() );
	}
}


export default Softmax;
