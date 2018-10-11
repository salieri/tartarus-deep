import { Activation } from './activation';
import { NDArray } from '../../ndarray';


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
