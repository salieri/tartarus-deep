import { Loss } from './loss';
import { Vector } from '../../math';

/**
 * L1
 */
export class L1 extends Loss
{
	calculate( yHat : Vector, y : Vector ) : number
	{
		// sum( abs( yHat - y ) )
		return yHat.sub( y ).abs().sum();
	}
}

