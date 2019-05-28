import { Session } from '../../../../src/nn/session';

describe(
  'Session',
  () => {
    it(
      'should provide a seedable randomizer',
      () => {
        const session = new Session('moo');

        session.getRandomizer().getSeed().should.equal('moo');

        session.random().should.be.a('number');
      },
    );
  },
);

