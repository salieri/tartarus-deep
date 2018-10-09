import Vector from '../../vector';

interface LossParams {
	[key : string]: any;
}


/**
 * A loss function returns a scalar measuring
 * the performance of a model. Inputs are:
 * predicted label (yHat) and actual label (y)
 */
class Loss
{
	protected params : LossParams;

	constructor( params : LossParams = {} )
	{
		this.params = params;
	}

	public calculate( yHat : Vector, y : Vector ) : number
	{
		throw new Error( 'Not implemented' );
	}


	public getDescriptor()
	{
		return {};
	}
}

export default Loss;
