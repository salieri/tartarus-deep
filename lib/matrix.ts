import NDArray from './ndarray';
import _ from 'lodash';

class Matrix extends NDArray
{
	constructor( ...dimensions : any[] )
	{
		super( ...dimensions );
	}


	protected validateConstructor( dimensions : any[] )
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
	public getCols() : number
	{
		return this.dimensions[ 1 ];
	}


	/**
	 * @public
	 * @returns {int}
	 */
	public getRows() : number
	{
		return this.dimensions[ 0 ];
	}


	/**
	 * @public
	 * @return {Matrix}
	 */
	public transpose() : Matrix
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
	public clone( targetObj : Matrix ) : Matrix
	{
		targetObj = targetObj || new Matrix( ...this.dimensions );

		return <Matrix>super.clone( targetObj );
	}
}


export default Matrix;

