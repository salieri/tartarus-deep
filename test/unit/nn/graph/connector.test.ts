import { DeferredCollection, Dense, Model } from '../../../../src/nn';
import { GraphNode, GraphProcessor, GraphProcessorDirection } from '../../../../src/nn/graph';
import { NDArray } from '../../../../src/math';

describe.only(
  'Graph Connector',
  () => {
    const model = new Model();
    const initialInput = new NDArray([0, 0.1]);

    const initialBackpropInput = new DeferredCollection(
      {
        loss: new NDArray([10000]),
        derivative: new NDArray([-1, -2]),
      },
    );


    it(
      'should prepare a test model',
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
        // const v3 = new NDArray([8]);

        d1.input.getDefault().get().should.equal(initialInput);

        d1.output.setDefaultValue(v1);
        d2.input.getDefault().get().should.equal(v1);

        d2.output.setDefaultValue(v2);
        d3.input.getDefault().get().should.equal(v2);
      },
    );


    it(
      'should connect raw backprop input and output feeds to each other',
      async () => {
        model.getGraph().assignBackpropInput(Model.coerceData(initialBackpropInput));

        const l1 = model.getGraph().find('layer-1');
        const l2 = model.getGraph().find('layer-2');
        const l3 = model.getGraph().find('layer-3');

        const d1 = l1.getEntity() as Dense;
        const d2 = l2.getEntity() as Dense;
        const d3 = l3.getEntity() as Dense;

        // const bv1 = new NDArray([-1, -2, -3]);
        const bv2 = new NDArray([-4, -5, -6, -7]);
        const bv3 = new NDArray([-8]);

        d3.rawBackpropInputs.getDefault().get('loss').get().should.equal(initialBackpropInput.get('loss').get());
        d3.rawBackpropInputs.getDefault().get('derivative').get().should.equal(initialBackpropInput.get('derivative').get());

        d3.backpropInput.getValue('loss').should.equal(initialBackpropInput.get('loss'));
        d3.backpropInput.getValue('derivative').should.equal(initialBackpropInput.get('derivative'));

        d3.backpropOutput.setValue('derivative', bv3);
        d2.backpropInput.getValue('derivative').should.equal(bv3);

        d2.backpropOutput.setValue('derivative', bv2);
        d1.backpropInput.getValue('derivative').should.equal(bv2);
      },
    );
  },
);
