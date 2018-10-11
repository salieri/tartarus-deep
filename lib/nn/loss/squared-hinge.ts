import { Loss, LossDescriptor } from './loss';
import { Vector } from '../../math';
import { NDArray } from '../../math';
import Joi from 'joi';


export class SquaredHinge extends Loss
{
	public calculate( yHat : Vector, y : Vector ) : number
	{
		return NDArray.iterate(
			( yHatVal : number, yVal : number ) : number => {
				return Math.pow( Math.max( 0, this.params.margin - yVal * yHatVal ), 2 );
			},
			yHat,
			y
		).mean();
	}


	public getDescriptor() : LossDescriptor
	{
		return {
			margin : Joi.number().optional().default( 1.0 )
		};
	}
}
