/* eslint-disable import/no-duplicates */

import _ from 'lodash';

import { ClassManager } from '../../../src/util';
import { Activation } from '../../../src/nn/activation';
import { Initializer } from '../../../src/nn/initializer';

// tslint:disable-next-line
import * as activations from '../../../src/nn/activation';

// tslint:disable-next-line
import * as initializers from '../../../src/nn/initializer';


describe(
  'Class Manager',
  () => {
    it(
      'should instantiate any class in the module',
      () => {
        const cm        = new ClassManager(activations, Activation);
        const activator = cm.factory('binary');

        activator.should.be.instanceOf(activations.Binary);
      },
    );


    it(
      'should instantiate a class with parameters',
      () => {
        const cm        = new ClassManager(activations, Activation);
        const activator = cm.factory('ReLU', undefined, { leak: 1.23 });

        activator.should.be.instanceOf(activations.ReLU);
        activator.params.leak.should.equal(1.23);
      },
    );


    it(
      'should return the instance, if one is passed to the coerce function',
      () => {
        const cm        = new ClassManager(activations, Activation);
        const activator = cm.coerce('bent-identity');

        activator.should.be.instanceOf(activations.BentIdentity);

        cm.coerce(activator).should.equal(activator);
        cm.coerce('bent-identity').should.not.equal(activator);
      },
    );


    it(
      'should detect all/only relevant classes',
      () => {
        const cm = new ClassManager(initializers, Initializer);

        _.keys(cm.getKnownClasses()).should.deep.equal(['constant', 'one', 'randomuniform', 'zero']);
      },
    );
  },
);
