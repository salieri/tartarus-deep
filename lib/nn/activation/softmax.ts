import { Activation } from './activation';
import { NDArray } from '../../math/ndarray';

/**
 * Softmax
 */
export class Softmax extends Activation
{
	public calculate( z : NDArray ) : NDArray
	{
		return z.exp().div( z.exp().sum() );
	}
}
