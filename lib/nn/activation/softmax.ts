import Activation from '.';
import NDArray from '../../ndarray';

/**
 * Softmax
 */
class Softmax extends Activation
{
	public calculate( z : NDArray ) : NDArray
	{
		return z.exp().div( z.exp().sum() );
	}
}


export default Softmax;
