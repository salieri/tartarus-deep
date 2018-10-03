
import _ from 'lodash';
import NDArray from './ndarray.mjs';
import Matrix from './matrix.mjs';


const Math = {

	elementwise : {
		/**
		 * Elementwise operation
		 * @param {NDArray|Matrix} a
		 * @param {NDArray|Matrix|Number} b
		 * @param {function(Number, Number)} operationCb
		 * @param {string} opName
		 * @return {NDArray|Matrix}
		 * @private
		 */
		baseOperation : ( a, b, operationCb, opName ) => {
			if( ( b instanceof NDArray ) || ( b instanceof Matrix ) )
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
		 * @param {NDArray|Matrix} a
		 * @param {NDArray|Number} b
		 * @return {NDArray|Matrix}
		 * @public
		 */
		add : ( a, b ) => {
			return Math.elementwise.baseOperation( a, b, ( aVal, bVal ) => ( aVal + bVal ), 'addition' );
		},


		/**
		 * Elementwise subtract
		 * @param {NDArray|Matrix} a
		 * @param {NDArray|Number} b
		 * @return {NDArray|Matrix}
		 * @public
		 */
		sub : ( a, b ) => {
			return Math.elementwise.baseOperation( a, b, ( aVal, bVal ) => ( aVal - bVal ), 'subtraction' );
		},


		/**
		 * Elementwise multiplication
		 * @param {NDArray|Matrix} a
		 * @param {NDArray|Matrix|Number} b
		 * @return {NDArray|Matrix}
		 * @public
		 */
		mul : ( a, b ) => {
			return Math.elementwise.baseOperation( a, b, ( aVal, bVal ) => ( aVal * bVal ), 'multiplication' );
		},


		/**
		 * Elementwise div
		 * @param {NDArray|Matrix} a
		 * @param {NDArray|Matrix|Number} b
		 * @return {NDArray|Matrix}
		 * @public
		 */
		div : ( a, b ) => {
			return Math.elementwise.baseOperation( a, b, ( aVal, bVal ) => ( aVal / bVal ), 'division' );
		}

	}


};


export default Math;


