import { NDArray, Vector, VectorDirection } from '../../../src/math';


describe(
  'Vector',
  () => {
    it(
      'should refuse to create vectors from data with an invalid shape',
      () => {
        const nd = new NDArray([[1, 2, 3, 4], [4, 3, 2, 1]]);

        (() => new Vector(nd)).should.Throw(/Vector must have exactly one data dimension/);
        (() => new Vector([[[1, 2], [2, 3]], [[2, 3], [4, 5]]])).should.Throw(/Vector must have exactly one data dimension/);
      },
    );


    it(
      'should calculate dot product between two vectors',
      () => {
        const v1Val = [2, 4, 6];
        const v2Val = [3, 5, 7];

        const v1 = new Vector([2, 4, 6]);
        const v2 = new Vector([3, 5, 7]);

        v1.dot(v2).should.equal(
          v1Val[0] * v2Val[0]
          + v1Val[1] * v2Val[1]
          + v1Val[2] * v2Val[2],
        );
      },
    );


    it(
      'should calculate the length of a vector',
      () => {
        const v = new Vector([3, 4, 5]);

        v.length().should.equal(Math.sqrt((3 ** 2) + (4 ** 2) + (5 ** 2)));
      },
    );


    it(
      'should expand a vector into a matrix',
      () => {
        const v = new Vector([3, 4, 5]);

        (() => v.expandToMatrix(2, 2, VectorDirection.Col)).should.Throw(/Vector does not fit the shape of the matrix/);
        (() => v.expandToMatrix(3, 4, VectorDirection.Col)).should.Throw(/Vector does not fit the shape of the matrix/);
        (() => v.expandToMatrix(4, 3, VectorDirection.Row)).should.Throw(/Vector does not fit the shape of the matrix/);


        const m1 = v.expandToMatrix(4, 3, VectorDirection.Col);

        m1.getAt([0, 0]).should.equal(3);
        m1.getAt([0, 1]).should.equal(4);
        m1.getAt([0, 2]).should.equal(5);
        m1.getAt([1, 0]).should.equal(3);
        m1.getAt([2, 1]).should.equal(4);
        m1.getAt([3, 2]).should.equal(5);


        const m2 = v.expandToMatrix(3, 4, VectorDirection.Row);

        m2.getAt([0, 0]).should.equal(3);
        m2.getAt([1, 0]).should.equal(4);
        m2.getAt([2, 0]).should.equal(5);
        m2.getAt([0, 1]).should.equal(3);
        m2.getAt([1, 1]).should.equal(4);
        m2.getAt([2, 1]).should.equal(5);
      },
    );
  },
);
