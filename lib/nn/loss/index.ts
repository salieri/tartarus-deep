import Vector from '../../vector';

/**
 * A loss function returns a scalar measuring
 * the performance of a model. Inputs are:
 * predicted label (yHat) and actual label (y)
 */
class Loss
{
	protected params : object;

	constructor( params : object = {} )
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
