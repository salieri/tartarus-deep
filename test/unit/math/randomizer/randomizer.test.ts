import _ from 'lodash';

import { Xoshiro128 } from '../../../../src/math/randomizer';


describe(
  'Randomizer',
  () => {
    it(
      'should always produce the same random values, given the same seed',
      () => {
        const r = new Xoshiro128('hello world');
        const r2 = new Xoshiro128('hello world');
        const rv = new Xoshiro128('hello world');

        rv.random().should.equal(0.8235286064445972);
        rv.floatBetween(1, 2).should.equal(1.1764695146121085);
        rv.intBetween(10, 20).should.equal(11);
        rv.floatRange(3).should.equal(2.646128630731255);
        rv.intRange(20).should.equal(12);

        r.random().should.equal(r2.random());
        r.floatBetween(1, 2).should.equal(r2.floatBetween(1, 2));
        r.intBetween(10, 20).should.equal(r2.intBetween(10, 20));
        r.intRange(20).should.equal(r2.intRange(20));
        r.floatRange(30).should.equal(r2.floatRange(30));
        r.getSeed().should.equal(r2.getSeed());
      },
    );


    it(
      'should produce different random values, given different seed',
      () => {
        const r = new Xoshiro128('hello world');
        const r2 = new Xoshiro128('world hello');

        r.random().should.not.equal(r2.random());
        r.floatBetween(1, 2).should.not.equal(r2.floatBetween(1, 2));
        r.intBetween(10, 20).should.not.equal(r2.intBetween(10, 20));
        r.intRange(20).should.not.equal(r2.intRange(20));
        r.floatRange(30).should.not.equal(r2.floatRange(30));
        r.getSeed().should.not.equal(r2.getSeed());
      },
    );


    it(
      'should use a pseudorandom seed, if none provided',
      () => {
        const r = new Xoshiro128();
        const r2 = new Xoshiro128();

        r.getSeed().should.not.equal(r2.getSeed());
      },
    );


    it(
      'should respect range limits',
      () => {
        const r = new Xoshiro128('hello');
        const totalRounds = 10000;

        _.times(
          totalRounds,
          () => {
            r.random().should.be.within(0, 0.9999999999999);
            r.floatBetween(10, 20).should.be.within(10, 19.9999999999999);
            r.intBetween(5, 7).should.be.within(5, 6.9999999999999);
            r.floatRange(30).should.be.within(0, 29.9999999999999);
            r.intRange(20).should.be.within(0, 19.9999999999999);
          },
        );
      },
    );


    it(
      'should distribute random values evenly',
      () => {
        const r = new Xoshiro128('hello');
        const range = 30;
        const stats = _.times(range, () => (0));
        const totalRounds = 100000;

        _.times(
          totalRounds,
          () => {
            const idx = r.intRange(range);

            stats[idx] += 1;
          },
        );


        // even distirbution?
        _.each(
          stats,
          (count: number) => {
            count.should.be.greaterThan(0);

            const base = 1 / range;
            const variance = base / 10;

            (count / totalRounds).should.be.within(base - variance, base + variance);
          },
        );
      },
    );
  },
);

