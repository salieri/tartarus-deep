import {
  NDArray,
  Vector,
  VectorDirection,
} from '../../../src/math';


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


    it(
      'should slice a vector',
      () => {
        const v = new Vector([11, 22, 33, 44, 55, 66, 77, 88]);

        const s = v.slice([2], 3);

        s.should.be.an.instanceOf(Vector);

        s.get().should.deep.equal([33, 44, 55]);
      },
    );


    it(
      'should test whether index is in top K',
      () => {
        const v = new Vector([1, 2, 3, -10, 7, 2]);

        v.inTopK(0, 2).should.equal(false);
        v.inTopK(1, 2).should.equal(false);
        v.inTopK(2, 2).should.equal(true);
        v.inTopK(3, 2).should.equal(false);
        v.inTopK(4, 2).should.equal(true);
        v.inTopK(5, 2).should.equal(false);
      },
    );

    it(
      'should select the top K values',
      () => {
        const v = new Vector([1, 2, 3, -10, 7, 2]);

        const result = v.topK(2);

        result[0].value.should.equal(7);
        result[0].index.should.equal(4);

        result[1].value.should.equal(3);
        result[1].index.should.equal(2);
      },
    );


    it(
      'should find the index of the lowest value',
      () => {
        const v = new Vector([1, 2, 3, -10, 7, 2]);

        v.argmin().should.deep.equal([3]);
      },
    );


    it(
      'should find the index of the highest value',
      () => {
        const v = new Vector([1, 2, 3, -10, 7, 2]);

        v.argmax().should.deep.equal([4]);
      },
    );


    it(
      'should print out a vector',
      () => {
        const v = new Vector([13, 19]);

        v.toString().should.match(/Vector#.*: 13,19/);
      },
    );
  },
);
