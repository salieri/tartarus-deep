import { Loss, LossDescriptor } from './loss';
import { Vector } from '../../vector';
import Joi from 'joi';


export class Huber extends Loss
{
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


	public getDescriptor() : LossDescriptor
	{
		return {
			delta : Joi.number().optional().default( 1.0 )
		};
	}
}

