import { DeferredCollection, Dense, Model } from '../../../../src/nn';
import { NDArray } from '../../../../src/math';

describe(
  'Graph Connector',
  () => {
    const model = new Model();
    const initialInput = new NDArray([0, 0.1]);

    const initialBackpropInput = new DeferredCollection(
      {
        weight: new NDArray([10000]),
        derivative: new NDArray([-1, -2]),
      },
    );


    before(
      async () => {
        model
          .input(2)
          .push(new Dense({ units: 3 }, 'layer-1'))
          .push(new Dense({ units: 4 }, 'layer-2'))
          .push(new Dense({ units: 2 }, 'layer-3'));

        await model.compile();

        model.getGraph().assignInput(Model.coerceData(initialInput));
      },
    );


    it(
      'should connect graph nodes to each other',
      async () => {
        const l1 = model.getGraph().find('layer-1');
        const l2 = model.getGraph().find('layer-2');
        const l3 = model.getGraph().find('layer-3');

        l1.getInputNodes().length.should.equal(0);
        l1.getOutputNodes().length.should.equal(1);
        l1.getOutputNodes()[0].should.equal(l2);

        l2.getInputNodes().length.should.equal(1);
        l2.getInputNodes()[0].should.equal(l1);
        l2.getOutputNodes().length.should.equal(1);
        l2.getOutputNodes()[0].should.equal(l3);

        l3.getInputNodes().length.should.equal(1);
        l3.getInputNodes()[0].should.equal(l2);
        l3.getOutputNodes().length.should.equal(0);
      },
    );


    it(
      'should connect raw input and output feeds to each other',
      async () => {
        const l1 = model.getGraph().find('layer-1');
        const l2 = model.getGraph().find('layer-2');
        const l3 = model.getGraph().find('layer-3');

        const d1 = l1.getEntity() as Dense;
        const d2 = l2.getEntity() as Dense;
        const d3 = l3.getEntity() as Dense;

        const v1 = new NDArray([1, 2, 3]);
        const v2 = new NDArray([4, 5, 6, 7]);

        // @ts-ignore
        d1.data.input.getDefault().get().should.equal(initialInput);

        d1.data.output.setDefaultValue(v1);

        // @ts-ignore
        d2.data.input.getDefault().get().should.equal(v1);

        d2.data.output.setDefaultValue(v2);

        // @ts-ignore
        d3.data.input.getDefault().get().should.equal(v2);
      },
    );
  },
);
