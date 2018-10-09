import Activation from '.';
import NDArray from '../../ndarray';


/**
 * Gaussian
 */
class Gaussian extends Activation
{
	public calculate( z : NDArray ) : NDArray
	{
		return z.pow( 2 ).neg().exp();
	}
}


export default Gaussian;
