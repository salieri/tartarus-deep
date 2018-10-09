import Activation from '.';
import NDArray from '../../ndarray';


/**
 * Binary step
 */
class Binary extends Activation
{
	public calculate( z : NDArray ) : NDArray
	{
		return z.apply( ( val : number ) : number => ( val < 0 ? 0 : 1 ) );
	}
}


export default Binary;
