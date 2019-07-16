import {
  Matrix,
  MatrixDirection,
  NDArray,
  Vector,
} from '../../../src/math';


describe(
  'Matrix',
  () => {
    it(
      'should transpose matrices',
      () => {
        const m = new Matrix(
          [
            [1, 2, 3, 4, 5],
            [9, 8, 7, 6, 5],
          ],
        );

        m.transpose().toJSON().should.deep.equal(
          [
            [1, 9],
            [2, 8],
            [3, 7],
            [4, 6],
            [5, 5],
          ],
        );
      },
    );


    it(
      'should not multiply matrices of incompatible size',
      () => {
        const m1 = new Matrix(
          [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
          ],
        );

        const m2 = new Matrix(
          [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
          ],
        );

        (() => m1.matmul(m2)).should.Throw(/Cannot multiply matrices where a.cols does not match b.rows/);
      },
    );


    it(
      'should multiply matrices',
      () => {
        const m1 = new Matrix(
          [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
          ],
        );

        const m2 = new Matrix(
          [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
          ],
        );

        const result = m1.matmul(m2);

        result.getRows().should.equal(2);
        result.getCols().should.equal(3);

        result.toJSON().should.deep.equal(
          [
            [48, 54, 60],
            [136, 158, 180],
          ],
        );
      },
    );


    it(
      'should refuse to create matrices from data with an invalid shape',
      () => {
        const nd = new NDArray([1, 2, 3, 4]);

        (() => new Matrix(nd)).should.Throw(/Matrix must have exactly two data dimensions/);
        (() => new Matrix([[[1, 2], [2, 3]], [[2, 3], [4, 5]]])).should.Throw(/Matrix must have exactly two data dimensions/);
      },
    );


    it(
      'should clone a matrix',
      () => {
        const values = [[1, 2, 3], [4, 5, 6]];
        const m = new Matrix(values);

        m.clone().get().should.deep.equal(values);
      },
    );


    it(
      'should pick values diagonally from a matrix',
      () => {
        const m = new Matrix(
          [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
          ],
        );

        m.pickDiagonal().get().should.deep.equal([1, 5, 9]);
      },
    );


    it(
      'should fail to pick values diagonally from a non-square matrix',
      () => {
        const m = new Matrix(
          [
            [1, 2, 3, 4],
            [4, 5, 6, 7],
            [7, 8, 9, 8],
          ],
        );

        (() => m.pickDiagonal()).should.Throw(/Cannot pick diagonally across a matrix which columns and rows are not equal size/);
      },
    );


    it(
      'should slice matrices',
      () => {
        const m = new Matrix(
          [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
          ],
        );

        m.slice(MatrixDirection.Horizontal, 0, 1).get().should.deep.equal([[1, 2, 3]]);
        m.slice(MatrixDirection.Horizontal, 0, 2).get().should.deep.equal([[1, 2, 3], [4, 5, 6]]);

        m.slice(MatrixDirection.Vertical, 1, 1).get().should.deep.equal([[2], [5], [8]]);
      },
    );


    it(
      'should calculate dot products with other matrices',
      () => {
        const m = new Matrix([[1, 2, 3]]);
        const m2 = new Matrix([[3], [2], [1]]);

        const result = m.dot(m2);

        result.should.be.instanceOf(Matrix);
      },
    );


    it(
      'should calculate dot products with vectors',
      () => {
        const m = new Matrix([[1, 2, 3]]);
        const v = new Vector([3, 2, 1]);

        const result = m.dot(v);

        result.should.be.instanceOf(Vector);
      },
    );


    it(
      'should print out a matrix',
      () => {
        const m = new Matrix([[15, 17]]);

        m.toString().should.match(/Matrix#.*: 15,17/);
      },
    );
  },
);
