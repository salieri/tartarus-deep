import Activation from '.';
import NDArray from '../../ndarray';
import Joi from 'joi';


/**
 * Inverse square root unit
 */
class ISRU extends Activation
{
	public calculate( z : NDArray ) : NDArray
	{
		// z / ( sqrt( 1 + alpha * z^2 )
		return z.div( z.pow( 2 ).mul( this.params.alpha ).add( 1 ).sqrt() );
	}


	public getDescriptor() : object
	{
		return {
			alpha : Joi.number().optional().default( 0.0 ).description( 'Multiplier' )
		}
	}
}


export default ISRU;
