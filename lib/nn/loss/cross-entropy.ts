import Loss from './loss';
import Vector from '../../vector';


class CrossEntropy extends Loss
{
	calculate( yHat : Vector, y : Vector ) : number
	{
		return -( y.dot( <Vector>yHat.log() ) );
	}
}


export default CrossEntropy;
