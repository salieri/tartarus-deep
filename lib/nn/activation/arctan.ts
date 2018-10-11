import { Activation } from './activation';
import { NDArray } from '../../math/ndarray';


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

