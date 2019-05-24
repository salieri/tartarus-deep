/* eslint-disable import/no-duplicates */

import _ from 'lodash';

import { ClassManager } from '../../../src/util';
import { Activation } from '../../../src/nn/activation';
import { Initializer } from '../../../src/nn/initializer';

// tslint:disable-next-line
import * as activations from '../../../src/nn/activation';

// tslint:disable-next-line
import * as initializers from '../../../src/nn/initializer';

class Moo {}
class Moo2 extends Moo {}
class Moo3 extends Moo2 {}
class Moo4 extends Moo3 {}
class Moo5 extends Moo4 {}
class Moo6 extends Moo5 {}
class Moo7 extends Moo6 {}

class Oom {}
class Oom2 extends Oom {}


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


    it(
      'should deal with unexpected data in prototype check',
      () => {
        ClassManager.hasPrototypeCalled(0, 'Moo').should.equal(false);
        ClassManager.hasPrototypeCalled('', 'Moo').should.equal(false);
        ClassManager.hasPrototypeCalled([], 'Moo').should.equal(false);
        ClassManager.hasPrototypeCalled(false, 'Moo').should.equal(false);
        ClassManager.hasPrototypeCalled(null, 'Moo').should.equal(false);
        ClassManager.hasPrototypeCalled({}, 'Moo').should.equal(false);
      },
    );


    it(
      'should detect inheritance in prototype check',
      () => {
        ClassManager.hasPrototypeCalled(new Moo6(), 'Moo').should.equal(true);
        ClassManager.hasPrototypeCalled(new Moo5(), 'Moo').should.equal(true);
        ClassManager.hasPrototypeCalled(new Moo4(), 'Moo').should.equal(true);
        ClassManager.hasPrototypeCalled(new Moo3(), 'Moo').should.equal(true);
        ClassManager.hasPrototypeCalled(new Moo2(), 'Moo').should.equal(true);
        ClassManager.hasPrototypeCalled(new Moo(), 'Moo').should.equal(true);

        ClassManager.hasPrototypeCalled(new Oom(), 'Moo').should.equal(false);
        ClassManager.hasPrototypeCalled(new Oom2(), 'Moo').should.equal(false);
      },
    );


    it(
      'should give up prototype check if there are too many levels of inheritance',
      () => {
        ClassManager.hasPrototypeCalled(new Moo7(), 'Moo').should.equal(false);
      },
    );
  },
);
