import Activation from '.';
import NDArray from '../../ndarray';


/**
 * Sinusoid
 */
class Sinusoid extends Activation
{
	public calculate( z : NDArray ) : NDArray
	{
		return z.sin();
	}
}


export default Sinusoid;
