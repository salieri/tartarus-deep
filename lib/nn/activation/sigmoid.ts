import Activation from '.';
import NDArray from '../../ndarray';


/**
 * Sigmoid / soft step / logistic
 */
class Sigmoid extends Activation
{
	public calculate( z : NDArray ) : NDArray
	{
		// 1 / ( 1 + e^-z )
		const one = z.set( 1.0 );

		return one.div( z.neg().exp().add( 1 ) );
	}
}


export default Sigmoid;
