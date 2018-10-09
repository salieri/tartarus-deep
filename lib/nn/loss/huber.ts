import Loss from './';
import Vector from '../../vector';
import Joi from "joi";


class Huber extends Loss
{
	constructor( params : object )
	{
		super( params );
	}


	calculate( yHat : Vector, y : Vector ) : number
	{
		return yHat.apply(
			( yHatVal : number, pos : number[] ) : number => {
				const yVal = y.getAt( pos );

				if( yVal - yHatVal < this.params.delta )
				{
					return 0.5 * Math.pow( ( yVal - yHatVal ), 2 );
				}

				return this.params.delta * ( yVal - yHatVal ) - 0.5 * this.params.delta;
			}
		).mean();
	}


	public getDescriptor() : object
	{
		return {
			delta : Joi.number().optional().default( 1.0 )
		};
	}
}


export default Huber;
