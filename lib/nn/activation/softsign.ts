import { Activation } from './activation';
import { NDArray } from '../../math/ndarray';


/**
 * Softsign
 */
export class Softsign extends Activation
{
	public calculate( z : NDArray ) : NDArray
	{
		// z / ( 1 + |z| )
		return z.div( z.abs().add( 1.0 ) );
	}
}
