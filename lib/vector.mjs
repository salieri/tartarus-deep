import NDArray from './ndarray.mjs';
import Matrix from './matrix.mjs';
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


	getSize()
	{
		return this.dimensions[ 0 ];
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


	/**
	 * Expand vector into a matrix of the specified size, cloning the context of the vector on each row or column
	 * @param {int} rows
	 * @param {int} cols
	 * @param {String} direction 'row' or 'col'
	 * @returns {Matrix}
	 */
	expandToMatrix( rows, cols, direction )
	{
		const result = new Matrix( rows, cols );

		if(
			( ( direction === 'row' ) && ( rows !== this.getSize() ) ) ||
			( ( direction === 'col' ) && ( cols !== this.getSize() ) ) ||
			( ( direction !== 'row' ) && ( direction !== 'col' ) )
		)
		{
			throw new Error( `Vector does not fit the shape of the matrix` );
		}

		let tPos;

		for( let y = 0; y < rows; y++ )
		{
			if( direction === 'row' )
			{
				tPos = y;
			}

			for( let x = 0; x < cols; x++ )
			{
				if( direction === 'col' )
				{
					tPos = x;
				}

				result.setAt( [ y, x ], this.getAt( [ tPos ] ) );
			}
		}

		return result;
	}
}


export default Vector;

