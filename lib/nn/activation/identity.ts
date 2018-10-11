import { Activation } from './activation';
import { NDArray } from '../../math/ndarray';


/**
 * Identity
 */
export class Identity extends Activation
{
	public calculate( z : NDArray ) : NDArray
	{
		return z;
	}
}

