import Activation from '.';
import NDArray from '../../ndarray';


/**
 * Identity
 */
class Identity extends Activation
{
	public calculate( z : NDArray ) : NDArray
	{
		return z;
	}
}


export default Identity;
