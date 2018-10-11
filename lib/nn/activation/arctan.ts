import { Activation } from './activation';
import { NDArray } from '../../ndarray';


/**
 * ArcTan
 */
export class ArcTan extends Activation
{
	public calculate( z : NDArray ) : NDArray
	{
		return z.atan();
	}
}

