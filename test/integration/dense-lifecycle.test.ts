import {
  Concat,
  Dense,
  Model,
  model,
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
          .push(new Dense({ units: 4 }, 'layer-1'))
          .push(new Dense({ units: 4 }, 'layer-2'))
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


    it(
      'should not compile a network, if no input dimension has been defined',
      async () => {
        const m = new Model();

        m.push(new Dense({ units: 4 }));
        m.push(new Dense({ units: 4 }));

        await m.compile().should.be.rejectedWith(/Could not resolve input for layer/);
      },
    );


    it(
      'should not compile a network, if no output nodes have been defined',
      async () => {
        const m = new Model();

        m.add(new Dense({ units: 4 }));
        m.add(new Dense({ units: 4 }));

        await m.compile().should.be.rejectedWith(/Model.*has no defined outputs/);
      },
    );


    it(
      'should not create a network with two layers having the same name',
      () => {
        const m = new Model();

        m.push(new Dense({ units: 1 }, 'layer'));

        (() => m.push(new Dense({ units: 1 }, 'layer'))).should.throw(/Duplicate entity name.*already exists in this graph/);
      },
    );


    it(
      'should not let the same layer to be added twice',
      () => {
        const m = new Model();
        const layer = new Dense({ units: 1 });

        m.push(layer);

        (() => m.push(layer)).should.Throw(/Instance.*already exists in this graph/);
      },
    );


    it(
      'should prevent circular networks from being created (adding)',
      () => {
        const m = new Model();

        const l1 = new Dense({ units: 1 });
        const l2 = new Dense({ units: 1 });
        const l3 = new Dense({ units: 1 });
        const l4 = new Dense({ units: 1 });

        m.add(l1);
        m.add(l2, l1);
        m.add(l3, l2);
        m.add(l4, l3);

        (() => m.add(l1, l4)).should.Throw(/Instance.*already exists in this graph/);
        (() => m.link(l3, l2)).should.Throw(/Circular graph of nodes detected/);
        (() => m.link(l1, l4)).should.not.Throw();
        (() => m.link(l4, l1)).should.Throw(/Circular graph of nodes detected/);
      },
    );


    it(
      'should allow a node to have multiple input sources',
      async () => {
        const m = new Model();

        const layer1 = new Dense({ units: 3 }, 'layer-1');
        const layer2 = new Dense({ units: 5 }, 'layer-2');

        m.input(2);
        m.push(layer1);
        m.push(layer2);

        const anotherLayer = new Concat({ order: ['layer-2', 'layer-1'] }, 'another');

        m.add(anotherLayer, ['layer-1', 'layer-2']);

        await m.compile();

        m.getRawOutputs().count().should.equal(1);


        const layer1Out = layer1.getRawOutputs();
        const layer2Out = layer2.getRawOutputs();
        const anotherOut = anotherLayer.getRawOutputs();

        const layer1OutSize = layer1Out.getDefault().getDefault().countElements();
        const layer2OutSize = layer2Out.getDefault().getDefault().countElements();
        const anotherOutSize = anotherOut.getDefault().getDefault().countElements();

        layer1Out.count().should.equal(1);
        layer1OutSize.should.equal(3);

        layer2Out.count().should.equal(1);
        layer2OutSize.should.equal(5);

        anotherOut.count().should.equal(1);
        anotherOutSize.should.equal(layer1OutSize + layer2OutSize);
      },
    );
  },
);
