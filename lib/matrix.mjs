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


	/**
	 * @public
	 * @returns {int}
	 */
	getCols()
	{
		return this.dimensions[ 1 ];
	}


	/**
	 * @public
	 * @returns {int}
	 */
	getRows()
	{
		return this.dimensions[ 0 ];
	}


	/**
	 * @public
	 * @return {Matrix}
	 */
	transpose()
	{
		const result = new Matrix( this.getCols(), this.getRows() );

		for( let y = 0; y < this.getRows(); y++ )
		{
			for( let x = 0; x < this.getCols(); x++ )
			{
				result.setAt( [ x, y ], this.getAt( [ y, x ] ) );
			}
		}

		return result;
	}


	/**
	 * Clone a matrix
	 * @param {Matrix} [targetObj=null]
	 * @returns {Matrix}
	 * @public
	 */
	clone( targetObj )
	{
		targetObj = targetObj || new Matrix( ...this.dimensions );

		return super.clone( targetObj );
	}
}


export default Matrix;

