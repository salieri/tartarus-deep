import Activation from '.';
import NDArray from '../../ndarray';
import Joi from 'joi';


/**
 * Sigmoid-weighted linear unit / SiLU / Swish
 */
class SiLU extends Activation
{
	public calculate( z : NDArray ) : NDArray
	{
		return z.set( 1.0 ).div( z.neg().exp().add( 1.0 ) ).mul( z );
	}
}


export default SiLU;
