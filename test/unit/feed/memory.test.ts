import _ from 'lodash';
import { Label, MemoryInputFeed, Sample } from '../../../src/feed';
import { DeferredCollection, DeferredInputCollection } from '../../../src/nn';
import { NDArray, NDArrayCollection } from '../../../src/math';

describe(
  'Memory Input Feed',
  () => {
    it(
      'should instantiate a memory input feed from a labeled data array',
      async () => {
        const data = [{ x: [1], y: [0] }, { x: [2], y: [1] }, { x: [3], y: [0] }];
        const m = new MemoryInputFeed(data);

        m.count().should.equal(data.length);

        for (let i = 0; i < data.length; i += 1) {
          // eslint-disable-next-line no-await-in-loop
          const record = await m.next();

          if (!record.label) {
            throw new Error(`Missing label for index ${i}`);
          }

          const y = record.label.raw.getDefaultValue();
          const x = record.sample.raw.getDefaultValue();

          y.get().should.deep.equal(data[i].y);
          x.get().should.deep.equal(data[i].x);
        }
      },
    );

    it(
      'should instantiate a memory input feed from a non-labeled data array',
      async () => {
        const data = [{ x: [1] }, { x: [2] }, { x: [3] }];
        const m = new MemoryInputFeed(data);

        m.count().should.equal(data.length);

        for (let i = 0; i < data.length; i += 1) {
          // eslint-disable-next-line no-await-in-loop
          const record = await m.next();

          if (record.label) {
            throw new Error('Extraneous label');
          }

          const x = record.sample.raw.getDefaultValue();

          x.get().should.deep.equal(data[i].x);
        }
      },
    );

    it(
      'should produce a non-labeled feed, if at least one of the data samples does not contain a label',
      async () => {
        const data = [{ x: [1], y: [0] }, { x: [2] }, { x: [3], y: [0] }];
        const m = new MemoryInputFeed(data);

        m.count().should.equal(data.length);

        for (let i = 0; i < data.length; i += 1) {
          // eslint-disable-next-line no-await-in-loop
          const record = await m.next();

          if (record.label) {
            throw new Error('Extraneous label');
          }

          const x = record.sample.raw.getDefaultValue();

          x.get().should.deep.equal(data[i].x);
        }
      },
    );

    it(
      'should instantiate a memory input feed from array of samples',
      async () => {
        const d1 = new DeferredCollection(new NDArray([1]));
        const d2 = new DeferredCollection(new NDArray([2]));

        const data: Sample[] = [
          {
            name: 'moo',
            raw: new DeferredInputCollection(d1),
          },
          {
            raw: new DeferredInputCollection(d2),
          },
        ];

        const m = new MemoryInputFeed(data);

        m.count().should.equal(data.length);

        for (let i = 0; i < data.length; i += 1) {
          // eslint-disable-next-line no-await-in-loop
          const record = await m.next();

          if ((data[i].name) && (!record.sample.name)) {
            throw new Error('Missing sample name');
          }

          if ((!data[i].name) && (record.sample.name)) {
            throw new Error('Extraneous sample name');
          }

          if (record.label) {
            throw new Error('Extraneous label data');
          }

          const x = record.sample.raw.getDefaultValue();

          x.get().should.deep.equal(data[i].raw.getDefaultValue().get());
        }
      },
    );

    it(
      'should instantiate a memory input feed from NDArrays',
      async () => {
        const data: NDArray[] = _.times(5, (n: number) => (new NDArray([n])));

        const m = new MemoryInputFeed(data);

        m.count().should.equal(data.length);

        for (let i = 0; i < data.length; i += 1) {
          // eslint-disable-next-line no-await-in-loop
          const record = await m.next();

          if (record.label) {
            throw new Error('Extraneous label data');
          }

          const x = record.sample.raw.getDefaultValue();

          x.get().should.deep.equal(data[i].get());
        }
      },
    );

    it(
      'should instantiate a memory input feed from raw number arrays',
      async () => {
        const data: number[][] = _.times(5, (n: number) => ([n]));

        const m = new MemoryInputFeed(data);

        m.count().should.equal(data.length);

        for (let i = 0; i < data.length; i += 1) {
          // eslint-disable-next-line no-await-in-loop
          const record = await m.next();
          const x = record.sample.raw.getDefaultValue();

          x.get().should.deep.equal(data[i]);
        }
      },
    );

    it(
      'should instantiate a memory input feed from NDArray collections',
      async () => {
        const data: NDArrayCollection[] = [
          {
            a: new NDArray([1, 2, 3]),
            b: new NDArray([3, 2, 1]),
          },
          {
            a: new NDArray([5, 6, 7]),
            b: new NDArray([7, 6, 5]),
          },
        ];

        const m = new MemoryInputFeed(data);

        m.count().should.equal(data.length);

        for (let i = 0; i < data.length; i += 1) {
          // eslint-disable-next-line no-await-in-loop
          const record = await m.next();

          const a = record.sample.raw.get('a').getDefaultValue();
          const b = record.sample.raw.get('b').getDefaultValue();

          a.get().should.deep.equal(data[i].a.get());
          b.get().should.deep.equal(data[i].b.get());
        }
      },
    );

    it(
      'should instantiate a memory input feed with separated label NDArray data',
      async () => {
        const data: NDArray[] = _.times(5, (n: number) => (new NDArray([n])));
        const labels: NDArray[] = _.times(5, (n: number) => (new NDArray([n * 2])));

        const m = new MemoryInputFeed(data, labels);

        m.count().should.equal(data.length);

        for (let i = 0; i < data.length; i += 1) {
          // eslint-disable-next-line no-await-in-loop
          const record = await m.next();

          if (!record.label) {
            throw new Error('Missing label data');
          }

          const x = record.sample.raw.getDefaultValue();
          const y = record.label.raw.getDefaultValue();

          x.get().should.deep.equal(data[i].get());
          y.get().should.deep.equal(labels[i].get());
        }
      },
    );

    it(
      'should not instantiate a memory input, if the label and sample counts are different',
      async () => {
        const data: NDArray[] = _.times(5, (n: number) => (new NDArray([n])));
        const labels: NDArray[] = _.times(3, (n: number) => (new NDArray([n * 2])));

        (() => new MemoryInputFeed(data, labels)).should.Throw(/Sample and label lengths must match/);
      },
    );

    it(
      'should instantiate a memory input feed with separated label record data',
      async () => {
        const data: NDArray[] = _.times(2, (n: number) => (new NDArray([n])));

        const l1 = new NDArray([1]);
        const l2 = new NDArray([2]);

        const labels: Label[] = [
          {
            raw: new DeferredInputCollection(new DeferredCollection(l1)),
          },
          {
            raw: new DeferredInputCollection(new DeferredCollection(l2)),
          },
        ];

        const m = new MemoryInputFeed(data, labels);

        m.count().should.equal(data.length);

        for (let i = 0; i < data.length; i += 1) {
          // eslint-disable-next-line no-await-in-loop
          const record = await m.next();

          if (!record.label) {
            throw new Error('Missing label data');
          }

          const x = record.sample.raw.getDefaultValue();
          const y = record.label.raw.getDefaultValue();

          x.get().should.deep.equal(data[i].get());
          y.get().should.deep.equal(labels[i].raw.getDefaultValue().get());
        }
      },
    );

    it(
      'should allow samples to be added to the feed',
      async () => {
        const m = new MemoryInputFeed();

        m.add(new NDArray([1101]));

        const r = await m.next();

        r.sample.raw.getDefaultValue().get().should.deep.equal([1101]);
      },
    );
  },
);
