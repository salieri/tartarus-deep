import _ from 'lodash';
import NDArray from './ndarray';
import Matrix from './matrix';
import Vector from './vector';
import {VectorDirection} from './vector';


const Math = {

	elementwise : {
		/**
		 * Elementwise operation between NDArrays of the same size
		 * @param {NDArray} a
		 * @param {NDArray|number} b
		 * @param {function(number, number) : number} operationCb
		 * @param {string} opName
		 * @return {NDArray}
		 * @private
		 */
		baseOperation : ( a : NDArray, b : NDArray | number, operationCb : Function, opName : string ) : NDArray => {
			if( b instanceof NDArray )
			{
				if( _.isEqual( a.getDims(), b.getDims() ) === false )
				{
					throw new Error( `Cannot do elementwise ${opName} on NDArrays with differing dimensions` );
				}
			}

			const aClone = a.clone();

			aClone.traverse(
				( val : number, pos : number[] ) : number => {
					let bVal = b;

					if( b instanceof NDArray )
					{
						bVal = b.getAt( pos );
					}

					return operationCb( val, bVal );
				}
			);

			return aClone;
		},


		/**
		 * Elementwise add
		 */
		add : ( a : NDArray, b : NDArray|number ) : NDArray => {
			return Math.elementwise.baseOperation(
				a,
				b,
				( aVal : number, bVal : number ) : number => ( aVal + bVal ),
				'addition'
			);
		},


		/**
		 * Elementwise subtract
		 */
		sub : ( a : NDArray, b : NDArray | number ) : NDArray => {
			return Math.elementwise.baseOperation(
				a,
				b,
				( aVal : number, bVal : number ) : number => ( aVal - bVal ),
				'subtraction'
			);
		},


		/**
		 * Elementwise multiplication
		 */
		mul : ( a : NDArray, b : NDArray ) : NDArray => {
			return Math.elementwise.baseOperation(
				a,
				b,
				( aVal : number, bVal : number ) : number => ( aVal * bVal ),
				'multiplication'
			);
		},


		/**
		 * Elementwise div
		 */
		div : ( a : NDArray, b : NDArray ) : NDArray => {
			return Math.elementwise.baseOperation(
				a,
				b,
				( aVal : number, bVal : number ) : number => ( aVal / bVal ),
				'division'
			);
		}

	},


	/**
	 * Matrix multiplication
	 */
	matmul( a : Matrix, b : Matrix ) : Matrix
	{
		const aCols : number = a.getCols(),
			aRows : number = a.getRows(),
			bCols : number = b.getCols(),
			bRows : number = b.getRows();

		if( aCols !== bRows )
		{
			throw new Error( `Cannot multiply matrices where a.cols does not match b.rows` );
		}

		const result : Matrix = new Matrix( bCols, aRows );

		for( let y : number = 0; y < bCols; y++ )
		{
			for( let x : number = 0; x < aRows; x++ )
			{
				let val : number = 0;

				for( let i : number = 0; i < aCols; i++ )
				{
					val += a.getAt( [ y, i ] ) * b.getAt( [ i, x ] );
				}

				result.setAt( [ y, x ], val );
			}
		}

		return result;
	},


	/**
	 * Multiply matrix `a` with vector `b` by expanding `b` into a matrix
	 */
	vecmul( a : Matrix, b : Vector, dimension : VectorDirection ) : Matrix
	{
		const bMat = b.expandToMatrix( a.getCols(), a.getRows(), dimension );

		return Math.matmul( a, bMat );
	}

};


export default Math;


