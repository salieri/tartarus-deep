
import NDArray from './ndarray.mjs';
import _ from 'lodash';

class Matrix extends NDArray
{

	constructor( ...dimensions )
	{
		super( ...dimensions );
	}


	validateConstructor( dimensions )
	{
		super.validateConstructor( dimensions );

		if( dimensions.length === 1 )
		{
			const dimEl = dimensions[ 0 ];

			if( dimEl instanceof NDArray )
			{
				if( dimEl.countDims() !== 2 )
				{
					throw new Error( `Matrix must have exactly two data dimensions` );
				}
			}
			else if( _.isArray( dimEl ) === true )
			{
				if( dimEl.length !== 2 )
				{
					throw new Error( `Matrix must have exactly two data dimensions` );
				}
			}
		}
	}

}


export default Matrix;

