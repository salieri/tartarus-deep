import { DeferredCollection, DeferredValue } from '../../../../../src/nn/symbols';
import { NDArray } from '../../../../../src/math';
import { KeyNotFoundError } from '../../../../../src/error';

describe(
  'Deferred Collection',
  () => {
    it(
      'should declare, set, and get values',
      () => {
        const c = new DeferredCollection();
        const testVal = new NDArray([1, 2, 3]);

        c.declare('test', testVal.getDims());
        c.setValue('test', testVal);

        c.getValue('test').should.equal(testVal);

        c.get('test').should.be.an.instanceOf(DeferredValue);

        c.get('test').getDims().should.deep.equal(testVal.getDims());
        c.get('test').get().get().should.deep.equal(testVal.get());
      },
    );


    it(
      'should not let the same value to be declared multiple times',
      () => {
        const c = new DeferredCollection();
        const testVal = new NDArray([1, 2, 3]);

        c.declare('test', testVal.getDims());

        (() => c.declare('test', testVal.getDims())).should.Throw(/Key .* has already been declared/);
      },
    );


    it(
      'should return all declared keys',
      () => {
        const c = new DeferredCollection();

        c.declare('a', 1);
        c.declare('b', 2);

        c.declareDefault(3);

        c.getKeys().sort().should.deep.equal(['a', 'b', c.getDefaultKey()].sort());
      },
    );


    it(
      'should allow default value to be set',
      () => {
        const c = new DeferredCollection();

        (() => c.setDefaultValue(new NDArray(1))).should.Throw(KeyNotFoundError);
        (() => c.getDefaultValue()).should.Throw(KeyNotFoundError);
        (() => c.requireDefault()).should.Throw(KeyNotFoundError);

        c.declareDefault(1);

        c.setDefaultValue(new NDArray(1));

        (() => c.getDefaultValue()).should.not.Throw();
        (() => c.requireDefault()).should.not.Throw();
      },
    );


    it(
      'should allow default value to be mapped to any declared key',
      () => {
        const c = new DeferredCollection();

        c.declare('a', 1);
        c.declare('b', 1);

        (() => c.getDefaultValue()).should.Throw(KeyNotFoundError);

        const val = new NDArray([10]);

        c.setValue('b', val);

        c.setDefaultKey('b');

        c.getDefaultValue().should.equal(val);
      },
    );


    it(
      'should declare and set default value, if an NDArray is passed in constructor',
      () => {
        const val = new NDArray([1]);
        const c = new DeferredCollection(val);

        (() => c.getDefaultValue()).should.not.Throw();
        (() => c.requireDefault()).should.not.Throw();
        (() => c.getDefaultKey()).should.not.Throw();

        c.getDefaultValue().should.equal(val);
      },
    );
  },
);
