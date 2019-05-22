import {
  Dense,
  Model,
  model,
  symbols,
} from '../../src';

describe(
  'Dense Layer Network',
  () => {
    it(
      'should compile a network',
      async () => {
        const m = new Model();
        const lastLayer = new Dense({ units: 2 }, 'last-layer');

        m.input(1)
          .push(new Dense({ units: 4 }))
          .push(new Dense({ units: 4 }))
          .push(lastLayer);

        await m.compile();

        m.getState().should.equal(model.ModelState.Compiled);

        const outputs = m.getRawOutputs();

        outputs.count().should.equal(1);

        // @ts-ignore: TS24341
        outputs.getDefault().collection.should.equal(lastLayer.output);

        const out = outputs.getDefault().getDefault();

        out.getDims().should.deep.equal([2]);

        const outputNodes = m.getOutputNodes();

        outputNodes.length.should.equal(1);

        outputNodes[0].getName().should.equal('last-layer');
      },
    );


    it.skip(
      'should not compile a network, if no input dimension has been defined',
      async () => {
        const m = new Model();

        m.push(new Dense({ units: 4 }));

        (async () => (m.compile())).should.Throw(symbols.KeyNotFoundError);
      },
    );


    it.skip(
      'should not create a network with two layers having the same name',
      () => {
        const m = new Model();

        m.push(new Dense({ units: 1 }, 'layer'));

        (() => m.push(new Dense({ units: 1 }, 'layer'))).should.Throw(/moo/);
      },
    );


    it.skip(
      'should not let the same layer to be added twice',
      () => {
        const m = new Model();
        const layer = new Dense({ units: 1 });

        m.push(layer);

        (() => m.push(layer)).should.Throw(/moo/);
      },
    );
  },
);
