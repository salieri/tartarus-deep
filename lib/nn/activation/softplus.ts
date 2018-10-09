import Activation from '.';
import NDArray from '../../ndarray';


/**
 * SoftPlus
 */
class Softplus extends Activation
{
	public calculate( z : NDArray ) : NDArray
	{
		// log( 1 + e^z )
		return z.exp().add( 1 ).log();
	}
}


export default Softplus;
