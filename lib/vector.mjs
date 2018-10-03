import NDArray from './ndarray.mjs';
import _ from 'lodash';

class Vector extends NDArray
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
				if( dimEl.countDims() !== 1 )
				{
					throw new Error( `Vector must have exactly one data dimension` );
				}
			}
			else if( _.isArray( dimEl ) === true )
			{
				if( dimEl.length !== 1 )
				{
					throw new Error( `Vector must have exactly one data dimension` );
				}
			}
		}
	}


	/**
	 * Clone a vector
	 * @param {Vector} [targetObj=null]
	 * @returns {Vector}
	 * @public
	 */
	clone( targetObj )
	{
		targetObj = targetObj || new Vector( ...this.dimensions );

		return super.clone( targetObj );
	}
}


export default Vector;

