import { Activation } from './activation';
import { NDArray } from '../../math/ndarray';


/**
 * Sinusoid
 */
export class Sinusoid extends Activation
{
	public calculate( z : NDArray ) : NDArray
	{
		return z.sin();
	}
}
