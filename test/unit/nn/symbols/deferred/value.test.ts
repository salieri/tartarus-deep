import { DeferredValue } from '../../../../../src/nn/symbols';
import { InvalidValueError, ValueNotDeclaredError, ValueNotSetError } from '../../../../../src/error';
import { NDArray } from '../../../../../src/math';

describe(
  'Deferred Value',
  () => {
    it(
      'should require value to be declared before it can be accessed',
      () => {
        const value = new DeferredValue();

        (() => value.set(new NDArray([1, 2]))).should.Throw(ValueNotDeclaredError);
        (() => value.get()).should.Throw(ValueNotDeclaredError);
        (() => value.getDims()).should.Throw(ValueNotDeclaredError);
        (() => value.countElements()).should.Throw(ValueNotDeclaredError);

        value.declare(2);

        (() => value.getDims()).should.not.Throw();
        (() => value.countElements()).should.not.Throw();
        (() => value.get()).should.Throw(ValueNotSetError);

        value.set(new NDArray([1, 2]));

        (() => value.get()).should.not.Throw();
        (() => value.getDims()).should.not.Throw();
        (() => value.countElements()).should.not.Throw();
      },
    );


    it(
      'should throw at get, if value has not been set',
      () => {
        const value = new DeferredValue();

        value.declare(1);

        (() => value.get()).should.Throw(ValueNotSetError);

        value.set(new NDArray([123]));

        const nd = value.get();

        nd.getDims().should.deep.equal([1]);

        nd.getAt([0]).should.equal(123);
      },
    );


    it(
      'should reject setting value to something that does not match expected dimensions',
      () => {
        const value = new DeferredValue(2);

        (() => value.set(new NDArray([1, 2, 3]))).should.Throw(InvalidValueError);
        (() => value.set(new NDArray([[1, 2], [3, 4]]))).should.Throw(InvalidValueError);
        (() => value.set(new NDArray(1))).should.Throw(InvalidValueError);

        (() => value.set(new NDArray([1, 2]))).should.not.Throw();
      },
    );


    it(
      'should prevent declare() from being used multiple times',
      () => {
        const value = new DeferredValue(2);

        (() => value.declare(3)).should.throw(/Value dimensions have already been declared/);

        const v2 = new DeferredValue();

        v2.declare([1, 2]);

        (() => v2.declare(3)).should.Throw(/Value dimensions have already been declared/);
      },
    );
  },
);
