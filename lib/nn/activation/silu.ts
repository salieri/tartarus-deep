import { Activation } from './activation';
import { NDArray } from '../../math';

/**
 * Sigmoid-weighted linear unit / SiLU / Swish
 */
export class SiLU extends Activation
{
	public calculate( z : NDArray ) : NDArray
	{
		return z.set( 1.0 ).div( z.neg().exp().add( 1.0 ) ).mul( z );
	}
}
