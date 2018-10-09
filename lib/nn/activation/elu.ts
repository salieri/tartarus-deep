import Activation from '.';
import NDArray from '../../ndarray';
import Joi from "joi";


/**
 * Exponential linear unit
 */
class ELU extends Activation
{
	public calculate( z : NDArray ) : NDArray
	{
		return z.apply( ( val : number ) : number => ( val < 0 ? this.params.leak * ( Math.exp( val ) - 1 ) : val ) );
	}


	public getDescriptor() : object
	{
		return {
			leak : Joi.number().optional().default( 0.0 ).description( 'Leak multiplier' )
		}
	}
}


export default ELU;
