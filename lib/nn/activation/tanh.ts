import Activation from '.';
import NDArray from '../../ndarray';


/**
 * TanH
 */
class TanH extends Activation
{
	public calculate( z : NDArray ) : NDArray
	{
		// (e^z - e^-z) / (e^z + e^-z)
		return z.exp().sub( z.neg().exp() ).div( z.exp().add( z.neg().exp() ) );
	}
}


export default TanH;
