import Loss from './';
import Vector from '../../vector';
import NDArray from '../../ndarray';
import Joi from 'joi';


class SquaredHinge extends Loss
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


	public getDescriptor() : object
	{
		return {
			margin : Joi.number().optional().default( 1.0 )
		};
	}
}


export default SquaredHinge;
