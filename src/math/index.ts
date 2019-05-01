export * from './matrix';
export * from './ndarray';
export * from './randomizer';
export * from './vector';

/**
 * Multiply matrix `a` with vector `b` by expanding `b` into a matrix
 */

/* vecmul( a : Matrix, b : Vector, dimension : VectorDirection ) : Matrix
  {
    const bMat = b.expandToMatrix( a.getCols(), a.getRows(), dimension );

    return Index.matmul( a, bMat );
  }
*/

