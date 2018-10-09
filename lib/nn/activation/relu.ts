import Activation from '.';
import NDArray from '../../ndarray';
import Joi from 'joi';


/**
 * Rectified linear unit / leaky rectified linear unit / parameteric rectified linear unit
 */
class ReLU extends Activation
{

	public calculate( z : NDArray ) : NDArray
	{
		return z.apply( ( val : number ) : number => ( val < 0 ? this.params.leak * val : val ) );
	}


	public getDescriptor() : object
	{
		return {
			leak : Joi.number().optional().default( 0.0 ).description( 'Leak multiplier' )
		}
	}
}


export default ReLU;
