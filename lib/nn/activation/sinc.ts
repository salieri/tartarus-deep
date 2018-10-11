import { Activation } from './activation';
import { NDArray } from '../../math/ndarray';


/**
 * Sinc
 */
export class Sinc extends Activation
{
	public calculate( z : NDArray ) : NDArray
	{
		return z.apply(
			( val : number ) : number => ( ( ( val === 0 ) || ( val === 0.0 ) ) ? 1.0 : Math.sin( val ) / val )
		);
	}
}
