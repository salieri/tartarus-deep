import { Initializer } from './initializer';
import { NDArray } from '../../math';


export class One extends Initializer {

	public initialize( data : NDArray ) : NDArray
	{
		return data.set( 1.0 );
	}

}
