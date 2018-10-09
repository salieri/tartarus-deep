import Activation from '.';
import NDArray from '../../ndarray';


/**
 * ArcTan
 */
class ArcTan extends Activation
{
	public calculate( z : NDArray ) : NDArray
	{
		return z.atan();
	}
}


export default ArcTan;
