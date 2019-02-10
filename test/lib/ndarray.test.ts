import 'mocha';
import { expect } from 'chai';
import { NDArray } from '../../src/math';


describe(
  'N-dimensional Array',
  () => {
    it(
      'should not create NDArray without specified dimensions',
      () => {
        (() => (new NDArray())).should.Throw(/Unspecified dimensions/);
      }
    );


    it(
      'should create NDArray from dimension spec',
      () => {
        const m = new NDArray(3, 4);

        m.countDims().should.equal(2);
        m.getDims()[0].should.equal(3);
        m.getDims()[1].should.equal(4);

        m.get().length.should.equal(3);
        (<number[]>m.get()[1]).length.should.equal(4);
      }
    );


    it(
      'should create NDArray directly from data spec',
      () => {
        const m = new NDArray(
          [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0]
          ]
        );

        m.countDims().should.equal(2);
        m.getDims()[0].should.equal(3);
        m.getDims()[1].should.equal(4);

        m.get().length.should.equal(3);
        (<number[]>m.get()[1]).length.should.equal(4);

        m.getAt([0, 0]).should.equal(0);
        m.getAt([1, 1]).should.equal(1);
        m.getAt([1, 2]).should.equal(1);
        m.getAt([2, 3]).should.equal(0);
      }
    );


    it(
      'should reject NDArrays with inconsistent data shape',
      () => {
        (() => (new NDArray(
          [
            [1, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0]
          ]
          ))
        ).should.Throw(/Inconsistent data size/);
      }
    );


    it(
      'should refuse to set data to NDArrays, if the data shape does not match',
      () => {
        const nd = new NDArray(2, 3);

        (() => (nd.setData(
          [
            [1, 2, 3]
          ]
        ))).should.Throw(/Inconsistent data size/);

        (() => (nd.setData(
          [
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3]
          ]
        ))).should.Throw(/Inconsistent data size/);

        (() => (nd.setData(
          [
            [1, 2],
            [1, 2],
            [1, 2]
          ]
        ))).should.Throw(/Inconsistent data size/);
      }
    );


    it(
      'should create NDArray from another NDArray',
      () => {
        const mSource = new NDArray(
          [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0]
          ]
        );

        const m = new NDArray(mSource);

        m.countDims().should.equal(2);
        m.getDims()[0].should.equal(3);
        m.getDims()[1].should.equal(4);

        m.get().length.should.equal(3);
        (<number[]>m.get()[1]).length.should.equal(4);

        m.getAt([0, 0]).should.equal(0);
        m.getAt([1, 1]).should.equal(1);
        m.getAt([1, 2]).should.equal(1);
        m.getAt([2, 3]).should.equal(0);
      }
    );


    it(
      'should be comparable with other NDArrays',
      () => {
        const m1 = new NDArray(
          [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0]
          ]
        );

        const m2 = new NDArray(
          [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0]
          ]
        );

        const n1 = new NDArray(
          [
            [3, 0, 0, 0],
            [0, 3, 3, 0],
            [0, 0, 0, 3]
          ]
        );

        m1.equals(m1).should.equal(true);
        m1.equals(m2).should.equal(true);
        m2.equals(m1).should.equal(true);

        m1.equals(n1).should.equal(false);
        n1.equals(n1).should.equal(true);
      }
    );


    it(
      'should do elementwise add operations between n-dimensional arrays and numbers',
      () => {
        const nd = new NDArray(
          [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
          ]
        );

        nd.add(1).toJSON().should.deep.equal(
          [
            [2, 3, 4],
            [5, 6, 7],
            [8, 9, 10]
          ]
        );
      }
    );


    it(
      'should do elementwise subtract operations between n-dimensional arrays and numbers',
      () => {
        const nd = new NDArray(
          [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
          ]
        );

        nd.sub(1).toJSON().should.deep.equal(
          [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]
          ]
        );
      }
    );


    it(
      'should do elementwise mul operations between n-dimensional arrays and numbers',
      () => {
        const nd = new NDArray(
          [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
          ]
        );

        nd.mul(2).toJSON().should.deep.equal(
          [
            [1 * 2, 2 * 2, 3 * 2],
            [4 * 2, 5 * 2, 6 * 2],
            [7 * 2, 8 * 2, 9 * 2]
          ]
        );
      }
    );


    it(
      'should do elementwise div operations between n-dimensional arrays and numbers',
      () => {
        const nd = new NDArray(
          [
            [2, 4, 6],
            [8, 10, 12],
            [14, 16, 18]
          ]
        );

        nd.div(2).toJSON().should.deep.equal(
          [
            [2 / 2, 4 / 2, 6 / 2],
            [8 / 2, 10 / 2, 12 / 2],
            [14 / 2, 16 / 2, 18 / 2]
          ]
        );
      }
    );


    it(
      'should do elementwise add operations between two n-dimensional arrays',
      () => {
        const nd = new NDArray(
          [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
          ]
        );

        const nd2 = new NDArray(
          [
            [2, 3, 4],
            [5, 6, 7],
            [8, 9, 10]
          ]
        );


        nd.add(nd2).toJSON().should.deep.equal(
          [
            [1 + 2, 2 + 3, 3 + 4],
            [4 + 5, 5 + 6, 6 + 7],
            [7 + 8, 8 + 9, 9 + 10]
          ]
        );
      }
    );


    it(
      'should do elementwise subtract operations between two n-dimensional arrays',
      () => {
        const nd = new NDArray(
          [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
          ]
        );

        const nd2 = new NDArray(
          [
            [2, 3, 4],
            [5, 6, 7],
            [8, 9, 10]
          ]
        );


        nd.sub(nd2).toJSON().should.deep.equal(
          [
            [1 - 2, 2 - 3, 3 - 4],
            [4 - 5, 5 - 6, 6 - 7],
            [7 - 8, 8 - 9, 9 - 10]
          ]
        );
      }
    );


    it(
      'should do elementwise mul operations between two n-dimensional arrays',
      () => {
        const nd = new NDArray(
          [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
          ]
        );

        const nd2 = new NDArray(
          [
            [2, 3, 4],
            [5, 6, 7],
            [8, 9, 10]
          ]
        );


        nd.mul(nd2).toJSON().should.deep.equal(
          [
            [1 * 2, 2 * 3, 3 * 4],
            [4 * 5, 5 * 6, 6 * 7],
            [7 * 8, 8 * 9, 9 * 10]
          ]
        );
      }
    );


    it(
      'should do elementwise div operations between two n-dimensional arrays',
      () => {
        const nd = new NDArray(
          [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
          ]
        );

        const nd2 = new NDArray(
          [
            [2, 3, 4],
            [5, 6, 7],
            [8, 9, 10]
          ]
        );


        nd.div(nd2).toJSON().should.deep.equal(
          [
            [1 / 2, 2 / 3, 3 / 4],
            [4 / 5, 5 / 6, 6 / 7],
            [7 / 8, 8 / 9, 9 / 10]
          ]
        );
      }
    );


    it(
      'should not do elementwise operations between two n-dimensional arrays of different shape',
      () => {
        const nd1 = new NDArray(
          [
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3]
          ]
        );

        const nd2 = new NDArray(
          [
            [1, 2],
            [1, 2]
          ]
        );

        (() => nd1.add(nd2)).should.Throw(/Cannot do elementwise addition on NDArrays with differing dimensions/);
      }
    );

  }
);

