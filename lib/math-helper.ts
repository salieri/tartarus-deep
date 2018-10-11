import _ from 'lodash';
import { NDArray } from './ndarray';
import { Matrix } from './matrix';
import { Vector } from './vector';
import {VectorDirection} from './vector';


const MathHelper = {

	/**
	 * Multiply matrix `a` with vector `b` by expanding `b` into a matrix
	 */
/*	vecmul( a : Matrix, b : Vector, dimension : VectorDirection ) : Matrix
	{
		const bMat = b.expandToMatrix( a.getCols(), a.getRows(), dimension );

		return MathHelper.matmul( a, bMat );
	}*/

};


export default MathHelper;


