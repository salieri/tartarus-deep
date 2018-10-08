import Vector from '../../vector';

/**
 * A loss function returns a scalar measuring
 * the performance of a model. Inputs are:
 * predicted label (yHat) and actual label (y)
 */
class Loss
{
	constructor()
	{
	}

	calculate( yHat : Vector, y : Vector ) : number
	{
		throw new Error( 'Not implemented' );
	}
}

export default Loss;
