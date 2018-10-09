import Activation from '.';
import NDArray from '../../ndarray';


/**
 * Softsign
 */
class Softsign extends Activation
{
	public calculate( z : NDArray ) : NDArray
	{
		// z / ( 1 + |z| )
		return z.div( z.abs().add( 1.0 ) );
	}
}


export default Softsign;
