import Activation from '.';
import NDArray from '../../ndarray';
import Joi from 'joi';


/**
 * Bent identity
 */
class BentIdentity extends Activation
{
	public calculate( z : NDArray ) : NDArray
	{
		// ( sqrt( z^2 + 1 ) - 1 ) / 2 + z
		return z.pow( 2.0 ).add( 1.0 ).sqrt().sub( 1.0 ).div( 2.0 ).add( z );
	}
}


export default BentIdentity;
