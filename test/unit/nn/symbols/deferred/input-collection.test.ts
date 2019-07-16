import { DeferredCollection, DeferredInputCollection } from '../../../../../src/nn/symbols';
import { NDArray } from '../../../../../src/math';
import { KeyNotFoundError } from '../../../../../src/error';

function getSample(): DeferredInputCollection {
  const di = new DeferredInputCollection();

  const v1 = new DeferredCollection(new NDArray([1]));
  const v2 = new DeferredCollection(new NDArray([3]));

  di.set('test', v1);
  di.set('another', v2);

  return di;
}


describe(
  'Deferred Input Collection',
  () => {
    it(
      'should pick first input, if available',
      () => {
        const di = new DeferredInputCollection();

        (() => di.first()).should.Throw(/No inputs available/);

        const v1 = new DeferredCollection(new NDArray([1]));
        const v2 = new DeferredCollection(new NDArray([3]));

        di.set('test', v1);

        di.first().getCollection().should.equal(v1);

        di.set('another', v2);

        di.first().getCollection().should.equal(v1);
      },
    );

    it(
      'should unset values throughout',
      () => {
        const di = getSample();

        di.areAllDeclared().should.equal(true);
        di.areAllSet().should.equal(true);
        (() => di.snapshot()).should.not.Throw();

        di.unsetValues();

        di.areAllDeclared().should.equal(true);
        di.areAllSet().should.equal(false);

        (() => di.snapshot()).should.Throw(/Cannot snapshot -- some declared values in the input collection are not set/);
      },
    );


    it(
      'should filter inputs',
      () => {
        const di = getSample();

        const diFiltered = di.filter(['another']);

        diFiltered.getKeys().should.deep.equal(['another']);
      },
    );


    it(
      'should fail if filtered inputs do not exist',
      () => {
        const di = getSample();

        (() => di.filter(['moo'])).should.Throw(KeyNotFoundError);
        (() => di.filter(['moo'], true)).should.not.Throw();
      },
    );
  },
);
