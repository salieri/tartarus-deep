/* eslint-disable import/no-duplicates */

import { expect } from 'chai';

import { JoiEx, ClassManager } from '../../src/util';
import { Activation } from '../../src/nn/activation';

// tslint:disable-next-line
import * as activations from '../../src/nn/activation';


describe(
  'JoiEx',
  () => {
    it(
      'should pass instantiated objects through coercing validators',
      () => {
        const cm          = new ClassManager(activations, Activation);
        const schema      = JoiEx.activation();
        const activation  = cm.factory('sigmoid');
        const result      = JoiEx.validate(activation, schema);

        expect(result.error).to.equal(null);
        expect(result.value).to.equal(activation);
        expect(result.value).to.be.instanceOf(Activation);
        expect(result.value).to.be.instanceOf(activations.Sigmoid);
      },
    );

    it(
      'should not pass instantiated objects through coercing validators',
      () => {
        const schema      = JoiEx.activation();
        const activation  = { moo: 'moo' };
        const result      = JoiEx.validate(activation, schema);

        expect(result.error).to.match(/Unexpected data type passed to the coerce function/);
      },
    );


    it(
      'should instantiate class names passed to coerce() validator',
      () => {
        const schema  = JoiEx.activation();
        const result  = JoiEx.validate('sigmoid', schema);

        expect(result.error).to.equal(null);
        expect(result.value).to.be.instanceOf(Activation);
        expect(result.value).to.be.instanceOf(activations.Sigmoid);
      },
    );


    it(
      'should not instantiate unknown class names passed to coerce() validator',
      () => {
        const schema  = JoiEx.activation();
        const result  = JoiEx.validate('this-does-not-exist', schema);

        expect(result.error).to.match(/Unexpected data type passed to the coerce function/);
      },
    );


    it(
      'should instantiate default value, if no data passed',
      () => {
        const schema  = JoiEx.activation().optional().default('sigmoid');
        const result  = JoiEx.validate(undefined, schema);

        expect(result.error).to.equal(null);
        expect(result.value).to.be.instanceOf(Activation);
        expect(result.value).to.be.instanceOf(activations.Sigmoid);
      },
    );
  },
);
