import { Concat } from '../../../../src';

describe(
  'Layer',
  () => {
    it(
      'should not allow space or period to be used in layer name',
      () => {
        (() => (new Concat({}, 'hello.world'))).should.Throw(/Layer names may not contain spaces or periods/);
        (() => (new Concat({}, 'hello world'))).should.Throw(/Layer names may not contain spaces or periods/);
      },
    );


    it(
      'should reject invalid parameters',
      () => {
        (() => (new Concat({ moo: 2 } as any))).should.Throw(/"moo" is not allowed/);
      },
    );
  },
);
