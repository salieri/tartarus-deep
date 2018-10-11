import { Activation } from './activation';
import { NDArray } from '../../math';


/**
 * SoftPlus
 */
export class Softplus extends Activation
{
	public calculate( z : NDArray ) : NDArray
	{
		// log( 1 + e^z )
		return z.exp().add( 1 ).log();
	}
}
