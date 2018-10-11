import { Activation } from './activation';
import { NDArray } from '../../math';


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

