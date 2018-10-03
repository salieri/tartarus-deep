
import _ from 'lodash';
import NDArray from './ndarray.mjs';
import Matrix from './matrix.mjs';


const Math = {

	elementwise : {
		/**
		 * Elementwise operation
		 * @param {NDArray} a
		 * @param {NDArray} b
		 * @param {function(Number, Number)} operationCb
		 * @param {string} opName
		 * @return {NDArray}
		 * @private
		 */
		baseOperation : ( a, b, operationCb, opName ) => {
			if( b instanceof NDArray )
			{
				if( _.equals( a.getDims(), b.getDims() ) === false )
				{
					throw new Error( `Cannot do elementwise ${opName} on NDArrays with differing dimensions` );
				}
			}

			const aClone = a.clone();

			aClone.traverse(
				( val, pos ) => {
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
		 * @param {NDArray} a
		 * @param {NDArray|Number} b
		 * @return {NDArray}
		 * @public
		 */
		add : ( a, b ) => {
			return Math.elementwise.baseOperation( a, b, ( aVal, bVal ) => ( aVal + bVal ), 'addition' );
		},


		/**
		 * Elementwise subtract
		 * @param {NDArray} a
		 * @param {NDArray|Number} b
		 * @return {NDArray}
		 * @public
		 */
		sub : ( a, b ) => {
			return Math.elementwise.baseOperation( a, b, ( aVal, bVal ) => ( aVal - bVal ), 'subtraction' );
		},


		/**
		 * Elementwise multiplication
		 * @param {NDArray} a
		 * @param {NDArray|Number} b
		 * @return {NDArray}
		 * @public
		 */
		mul : ( a, b ) => {
			return Math.elementwise.baseOperation( a, b, ( aVal, bVal ) => ( aVal * bVal ), 'multiplication' );
		},


		/**
		 * Elementwise div
		 * @param {NDArray} a
		 * @param {NDArray|Number} b
		 * @return {NDArray}
		 * @public
		 */
		div : ( a, b ) => {
			return Math.elementwise.baseOperation( a, b, ( aVal, bVal ) => ( aVal / bVal ), 'division' );
		}

	},


	/**
	 * Matrix multiplication
	 * @public
	 * @param {Matrix} a
	 * @param {Matrix} b
	 * @return {Matrix}
	 */
	matmul( a, b )
	{
		const aCols	= a.getCols(),
			aRows	= a.getRows(),
			bCols	= b.getCols(),
			bRows	= b.getRows();

		if( aCols !== bRows )
		{
			throw new Error( `Cannot multiply matrices where a.cols does not match b.rows` );
		}

		const result = new Matrix( bCols, aRows );

		for( let y = 0; y < bCols; y++ )
		{
			for( let x = 0; x < aRows; x++ )
			{
				let val = 0;

				for( let i = 0; i < aCols; i++ )
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
	 * @param {Matrix} a
	 * @param {Vector} b
	 * @param {String} dimension 'row' or 'col'
	 * @return {Matrix}
	 * @public
	 */
	vecmul( a, b, dimension )
	{
		const bMat = b.expandToMatrix( a.getCols(), a.getRows(), dimension );

		return Math.matmul( a, bMat );
	}

};


export default Math;


