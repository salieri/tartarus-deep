import { Activation } from './activation';
import { NDArray } from '../../ndarray';


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

