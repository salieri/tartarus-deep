import NDArray from './ndarray';
import Matrix from './matrix';
import _ from 'lodash';

export enum VectorDirection {
	Row,
	Col
}


class Vector extends NDArray
{
	constructor( ...dimensions : any[] )
	{
		super( ...dimensions );
	}


	protected validateConstructor( dimensions : any[] ) : void
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
				if( Vector.resolveDimensions( dimEl ).length !== 1 )
				{
					throw new Error( `Vector must have exactly one data dimension` );
				}
			}
		}
	}


	/**
	 * Get vector size
	 */
	public getSize() : number
	{
		return this.dimensions[ 0 ];
	}


	/**
	 * Clone a vector
	 * @param {Vector} [targetObj=null]
	 * @returns {Vector}
	 * @public
	 */
	public clone( targetObj: Vector ) : Vector
	{
		targetObj = targetObj || new Vector( ...this.dimensions );

		return <Vector>super.clone( targetObj );
	}


	/**
	 * Expand vector into a matrix of the specified size, cloning the context of the vector on each row or column
	 * @param {int} rows
	 * @param {int} cols
	 * @param {String} direction 'row' or 'col'
	 * @returns {Matrix}
	 */
	public expandToMatrix( rows : number, cols : number, direction : VectorDirection ) : Matrix
	{
		const result : Matrix = new Matrix( rows, cols );

		if(
			( ( direction === VectorDirection.Row ) && ( rows !== this.getSize() ) ) ||
			( ( direction === VectorDirection.Col ) && ( cols !== this.getSize() ) )
		)
		{
			throw new Error( `Vector does not fit the shape of the matrix` );
		}

		let tPos : number = 0;

		for( let y : number = 0; y < rows; y++ )
		{
			if( direction === VectorDirection.Row )
			{
				tPos = y;
			}

			for( let x : number = 0; x < cols; x++ )
			{
				if( direction === VectorDirection.Col )
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
