import { NDArray } from './ndarray';
import { Vector } from './vector';
import _ from 'lodash';


export class Matrix extends NDArray
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
				if( dimEl.countDims() !== 2 )
				{
					throw new Error( `Matrix must have exactly two data dimensions` );
				}
			}
			else if( _.isArray( dimEl ) === true )
			{
				if( Matrix.resolveDimensions( dimEl ).length !== 2 )
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
	 * @return { Matrix }
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
	 * @param { Matrix } [targetObj=null]
	 * @returns { Matrix }
	 * @public
	 */
	public clone( targetObj : Matrix ) : Matrix
	{
		targetObj = targetObj || new Matrix( ...this.dimensions );

		return <Matrix>super.clone( targetObj );
	}


	/**
	 * Matrix multiplication
	 */
	public matmul( b : Matrix ) : Matrix
	{
		const aCols : number = this.getCols(),
			aRows : number = this.getRows(),
			bCols : number = b.getCols(),
			bRows : number = b.getRows();

		if( aCols !== bRows )
		{
			throw new Error( `Cannot multiply matrices where a.cols does not match b.rows` );
		}

		const result : Matrix = new Matrix( aRows, bCols );

		for( let y : number = 0; y < aRows; y++ )
		{
			for( let x : number = 0; x < bCols; x++ )
			{
				let val : number = 0;

				for( let i : number = 0; i < aCols; i++ )
				{
					val += this.getAt( [ y, i ] ) * b.getAt( [ i, x ] );
				}

				result.setAt( [ y, x ], val );
			}
		}

		return result;
	}


	/**
	 * Multiply matrix by a vector
	 */
	public vecmul( b : Vector ) : Vector
	{
		const aCols : number = this.getCols(),
			aRows : number = this.getRows(),
			bSize : number = b.getSize();

		const result : Vector = new Vector( aRows );

		for( let y : number = 0; y < aRows; y++ )
		{
			let val : number = 0;

			for( let x : number = 0; x < bSize; x++ )
			{
				val += this.getAt( [ y, x ] ) * b.getAt( [ x ] );
			}

			result.setAt( [ y ], val );
		}

		return result;
	}
}


