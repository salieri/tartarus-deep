import _ from 'lodash';

import {
  NDArray,
  NDArrayPosition,
  NumberTreeElement,
  Xoshiro128,
} from '../../../src/math';


describe(
  'N-dimensional Array',
  () => {
    it(
      'should not create NDArray without specified dimensions',
      () => {
        (() => (new NDArray())).should.Throw(/Unspecified dimensions/);
      },
    );


    it(
      'should create NDArray from dimension spec',
      () => {
        const m = new NDArray(3, 4);

        m.countDims().should.equal(2);
        m.getDims()[0].should.equal(3);
        m.getDims()[1].should.equal(4);

        m.get().length.should.equal(3);
        (m.get()[1] as number[]).length.should.equal(4);
      },
    );


    it(
      'should create NDArray directly from data spec',
      () => {
        const m = new NDArray(
          [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
          ],
        );

        m.countDims().should.equal(2);
        m.getDims()[0].should.equal(3);
        m.getDims()[1].should.equal(4);

        m.get().length.should.equal(3);
        (m.get()[1] as number[]).length.should.equal(4);

        m.getAt([0, 0]).should.equal(0);
        m.getAt([1, 1]).should.equal(1);
        m.getAt([1, 2]).should.equal(1);
        m.getAt([2, 3]).should.equal(0);
      },
    );


    it(
      'should reject NDArrays with inconsistent data shape',
      () => {
        (() => (new NDArray(
          [
            [1, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0],
          ],
        ))
        ).should.Throw(/Inconsistent data size/);
      },
    );


    it(
      'should refuse to set data to NDArrays, if the data shape does not match',
      () => {
        const nd = new NDArray(2, 3);

        (() => (nd.setData(
          [
            [1, 2, 3],
          ],
        ))).should.Throw(/Inconsistent data size/);

        (() => (nd.setData(
          [
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
          ],
        ))).should.Throw(/Inconsistent data size/);

        (() => (nd.setData(
          [
            [1, 2],
            [1, 2],
            [1, 2],
          ],
        ))).should.Throw(/Inconsistent data size/);
      },
    );


    it(
      'should create NDArray from another NDArray',
      () => {
        const mSource = new NDArray(
          [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
          ],
        );

        const m = new NDArray(mSource);

        m.countDims().should.equal(2);
        m.getDims()[0].should.equal(3);
        m.getDims()[1].should.equal(4);

        m.get().length.should.equal(3);
        (m.get()[1] as number[]).length.should.equal(4);

        m.getAt([0, 0]).should.equal(0);
        m.getAt([1, 1]).should.equal(1);
        m.getAt([1, 2]).should.equal(1);
        m.getAt([2, 3]).should.equal(0);
      },
    );


    it(
      'should be comparable with other NDArrays',
      () => {
        const m1 = new NDArray(
          [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
          ],
        );

        const m2 = new NDArray(
          [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
          ],
        );

        const n1 = new NDArray(
          [
            [3, 0, 0, 0],
            [0, 3, 3, 0],
            [0, 0, 0, 3],
          ],
        );

        m1.equals(m1).should.equal(true);
        m1.equals(m2).should.equal(true);
        m2.equals(m1).should.equal(true);

        m1.equals(n1).should.equal(false);
        n1.equals(n1).should.equal(true);
      },
    );


    it(
      'should do elementwise add operations between n-dimensional arrays and numbers',
      () => {
        const nd = new NDArray(
          [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
          ],
        );

        nd.add(1).toJSON().should.deep.equal(
          [
            [2, 3, 4],
            [5, 6, 7],
            [8, 9, 10],
          ],
        );
      },
    );


    it(
      'should do elementwise subtract operations between n-dimensional arrays and numbers',
      () => {
        const nd = new NDArray(
          [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
          ],
        );

        nd.sub(1).toJSON().should.deep.equal(
          [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
          ],
        );
      },
    );


    it(
      'should do elementwise mul operations between n-dimensional arrays and numbers',
      () => {
        const nd = new NDArray(
          [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
          ],
        );

        nd.mul(2).toJSON().should.deep.equal(
          [
            [1 * 2, 2 * 2, 3 * 2],
            [4 * 2, 5 * 2, 6 * 2],
            [7 * 2, 8 * 2, 9 * 2],
          ],
        );
      },
    );


    it(
      'should do elementwise div operations between n-dimensional arrays and numbers',
      () => {
        const nd = new NDArray(
          [
            [2, 4, 6],
            [8, 10, 12],
            [14, 16, 18],
          ],
        );

        nd.div(2).toJSON().should.deep.equal(
          [
            [2 / 2, 4 / 2, 6 / 2],
            [8 / 2, 10 / 2, 12 / 2],
            [14 / 2, 16 / 2, 18 / 2],
          ],
        );
      },
    );


    it(
      'should do elementwise add operations between two n-dimensional arrays',
      () => {
        const nd = new NDArray(
          [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
          ],
        );

        const nd2 = new NDArray(
          [
            [2, 3, 4],
            [5, 6, 7],
            [8, 9, 10],
          ],
        );


        nd.add(nd2).toJSON().should.deep.equal(
          [
            [1 + 2, 2 + 3, 3 + 4],
            [4 + 5, 5 + 6, 6 + 7],
            [7 + 8, 8 + 9, 9 + 10],
          ],
        );
      },
    );


    it(
      'should do elementwise subtract operations between two n-dimensional arrays',
      () => {
        const nd = new NDArray(
          [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
          ],
        );

        const nd2 = new NDArray(
          [
            [2, 3, 4],
            [5, 6, 7],
            [8, 9, 10],
          ],
        );


        nd.sub(nd2).toJSON().should.deep.equal(
          [
            [1 - 2, 2 - 3, 3 - 4],
            [4 - 5, 5 - 6, 6 - 7],
            [7 - 8, 8 - 9, 9 - 10],
          ],
        );
      },
    );


    it(
      'should do elementwise mul operations between two n-dimensional arrays',
      () => {
        const nd = new NDArray(
          [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
          ],
        );

        const nd2 = new NDArray(
          [
            [2, 3, 4],
            [5, 6, 7],
            [8, 9, 10],
          ],
        );


        nd.mul(nd2).toJSON().should.deep.equal(
          [
            [1 * 2, 2 * 3, 3 * 4],
            [4 * 5, 5 * 6, 6 * 7],
            [7 * 8, 8 * 9, 9 * 10],
          ],
        );
      },
    );


    it(
      'should do elementwise div operations between two n-dimensional arrays',
      () => {
        const nd = new NDArray(
          [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
          ],
        );

        const nd2 = new NDArray(
          [
            [2, 3, 4],
            [5, 6, 7],
            [8, 9, 10],
          ],
        );


        nd.div(nd2).toJSON().should.deep.equal(
          [
            [1 / 2, 2 / 3, 3 / 4],
            [4 / 5, 5 / 6, 6 / 7],
            [7 / 8, 8 / 9, 9 / 10],
          ],
        );
      },
    );


    it(
      'should not do elementwise operations between two n-dimensional arrays of different shape',
      () => {
        const nd1 = new NDArray(
          [
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
          ],
        );

        const nd2 = new NDArray(
          [
            [1, 2],
            [1, 2],
          ],
        );

        (() => nd1.add(nd2)).should.Throw(/Cannot do elementwise addition on NDArrays with differing dimensions/);
      },
    );


    it(
      'should concatenate arrays of different shapes',
      () => {
        const nd1 = new NDArray(
          [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
          ],
        );

        const nd2 = new NDArray(
          [
            [9, 10],
            [11, 12],
            [13, 14],
          ],
        );

        const concat = nd1.concat(nd2);

        const dims = concat.getDims();

        dims.length.should.equal(1);
        dims[0].should.equal(8 + 6);

        _.times(
          8 + 6,
          (n: number) => {
            concat.getAt([n]).should.equal(n + 1);
          },
        );
      },
    );


    it(
      'should calculate a sum of its elements',
      () => {
        const nd = new NDArray([1, 2, 3]);

        nd.sum().should.equal(1 + 2 + 3);
      },
    );


    it(
      'should calculate a sum of its elements with a custom callback',
      () => {
        const nd = new NDArray([1, 2, 3]);

        nd.sum((n: number) => (n * 2)).should.equal((1 + 2 + 3) * 2);
      },
    );


    it(
      'should normalize an array',
      () => {
        const nd = new NDArray([1, 2, 3]);

        const norm = nd.normalize();

        const expSum = Math.sqrt(((1 ** 2) + (2 ** 2) + (3 ** 2)));

        norm.getAt([0]).should.equal(1 / expSum);
        norm.getAt([1]).should.equal(2 / expSum);
        norm.getAt([2]).should.equal(3 / expSum);
      },
    );


    it(
      'should calculate mean of an array',
      () => {
        const nd = new NDArray([1, 2, 3]);

        nd.mean().should.equal((1 + 2 + 3) / 3);
      },
    );


    it(
      'should iterate multiple arrays of the same size',
      () => {
        const nd1 = new NDArray([1, 1, 1, 1]);
        const nd2 = new NDArray([2, 2, 2, 2]);
        const nd3 = new NDArray([3, 3, 3, 3]);

        const result = nd1.iterate(
          (vals: number[]): number => ((vals[0] + vals[1]) * vals[2]),
          nd2,
          nd3,
        );

        result.countDims().should.equal(1);
        result.countElements().should.equal(4);

        result.traverse((val: number): void => { val.should.equal((1 + 2) * 3); });
      },
    );


    it(
      'should clamp values of an array',
      () => {
        const nd = new NDArray([-2, -1, 0, 0.5, 1, 2]);

        const result = nd.clamp(0, 1);

        result.get().should.deep.equal([0, 0, 0, 0.5, 1, 1]);
      },
    );


    it(
      'should do elementwise math operations',
      () => {
        const vals = [-1, 0, 1, 0.4, 0.2];
        const ops = ['sqrt', 'atan', 'acos', 'asin', 'tan', 'cos', 'sin', 'exp', 'log', 'abs'];

        _.each(
          ops,
          (opName: string) => {
            const nd = new NDArray(vals);
            const result = (nd as any)[opName]() as NDArray;

            _.each(
              result.get(),
              (resultVal: NumberTreeElement, index: number) => {
                const originalVal = vals[index];

                resultVal.should.deep.equal((Math as any)[opName](originalVal));
              },
            );
          },
        );
      },
    );


    it(
      'should negate values',
      () => {
        const nd = new NDArray([-1, 0, 1, 2]);

        const result = nd.neg();

        result.get().should.deep.equal([1, -0, -1, -2]);
      },
    );


    it(
      'should calculate elementwise exponents',
      () => {
        const nd = new NDArray([-1, 0, 1, 2]);

        const result = nd.pow(3);

        result.get().should.deep.equal([(-1) ** 3, 0 ** 3, 1 ** 3, 2 ** 3]);
      },
    );


    it(
      'should calculate elementwise exponents where exponents are provided by another array',
      () => {
        const nd = new NDArray([-2, 4, 8, 10]);
        const pow = new NDArray([3, 4, 5, 6]);

        const result = nd.pow(pow);

        result.get().should.deep.equal([(-2) ** 3, 4 ** 4, 8 ** 5, 10 ** 6]);
      },
    );


    it(
      'should not equal with an array of different dimensions',
      () => {
        const a = new NDArray([0, 1, 2]);
        const a2 = new NDArray([0, 1, 2]);
        const b = new NDArray([0, 1, 2, 3]);
        const c = new NDArray([0, 1, 2], [1, 2, 3]);

        a.equals(a).should.equal(true);
        a.equals(a2).should.equal(true);
        a.equals(b).should.equal(false);
        a.equals(c).should.equal(false);
      },
    );


    it(
      'should produce string representation of itself',
      () => {
        const vals = [0, 2, 4, 6, 8];
        const nd = new NDArray(vals);

        nd.toString().should.match(new RegExp(`${_.join(vals, ',')}`));
        nd.toString().should.match(/NDArray#/);
      },
    );


    it(
      'should zero all values',
      () => {
        const nd = new NDArray([1, 2, 3, 4]);

        nd.zero().get().should.deep.equal([0, 0, 0, 0]);
      },
    );


    it(
      'should randomize elements with a randomizer',
      () => {
        const r = new Xoshiro128('hello-world');
        const nd = new NDArray(5);

        const expected = [
          0.9411756654735655,
          0.47058743005618453,
          0.7647048090584576,
          0.47081434493884444,
          0.832484302809462,
        ];

        nd.rand(0, 1, r).get().should.deep.equal(expected);
      },
    );


    it(
      'should randomize elements without a randomizer',
      () => {
        const nd = new NDArray(5);

        // This is a super weak test
        nd.rand(0, 1).get().should.not.deep.equal([0, 0, 0, 0, 0]);
      },
    );


    it(
      'should traverse all arrays in the NDArray',
      () => {
        const nd = new NDArray([[1, 2, 3], [2, 3, 4], [4, 5, 6]]);

        let idx = 0;

        const expectedPos = [[0], [1], [2]];
        const expectedVal = [[1, 2, 3], [2, 3, 4], [4, 5, 6]];
        const expectedCount = 3;


        nd.traverseArrays(
          (branch: NumberTreeElement[], pos: NDArrayPosition) => {
            branch.should.deep.equal(expectedVal[idx]);
            pos.should.deep.equal(expectedPos[idx]);

            idx += 1;
          },
        );

        idx.should.equal(expectedCount);
      },
    );


    it(
      'should fail to set data on an NDArray, if the data has an invalid shape',
      () => {
        const nd = new NDArray([[0, 1, 2], [2, 3, 4]]);

        (() => nd.setData([1, 2, 3, 4])).should.Throw(/Inconsistent data size/);

        (() => nd.setData([[0, 1, 2], [3, 2, 1, 2]])).should.Throw(/Inconsistent data size/);
      },
    );


    it(
      'should validate position paths',
      () => {
        const nd = new NDArray([[0, 1, 2], [2, 3, 4]]);

        // @ts-ignore
        (() => nd.validatePosition([0])).should.Throw(/Invalid position path: expected .* dimensions/);

        // @ts-ignore
        (() => nd.validatePosition([0, 3])).should.Throw(/Invalid position path: Dimension .* position should be/);

        // @ts-ignore
        (() => nd.validatePosition([2, 1])).should.Throw(/Invalid position path: Dimension .* position should be/);

        // @ts-ignore
        (() => nd.validatePosition([0, 0])).should.not.Throw();
      },
    );
  },
);

