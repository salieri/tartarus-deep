import Loss from '.';
import {LossDescriptor} from '.';
import Vector from '../../vector';
import NDArray from '../../ndarray';
import Joi from 'joi';


class Hinge extends Loss
{
	public calculate( yHat : Vector, y : Vector ) : number
	{
		return NDArray.iterate(
			( yHatVal : number, yVal : number ) : number => {
				return Math.max( 0, this.params.margin - yVal * yHatVal );
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


export default Hinge;
