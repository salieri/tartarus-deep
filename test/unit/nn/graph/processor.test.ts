import { Dense, Model } from '../../../../src/nn';
import { GraphNode, GraphProcessor, GraphProcessorDirection } from '../../../../src/nn/graph';

describe(
  'Node Processor',
  () => {
    const model = new Model();

    it(
      'should prepare a test model',
      async () => {
        model
          .input(2)
          .push(new Dense({ units: 3 }, 'layer-1'))
          .push(new Dense({ units: 4 }, 'layer-2'))
          .push(new Dense({ units: 1 }, 'layer-3'));

        await model.compile();

        model.getGraph().assignInput(Model.coerceData([1, 2]));
      },
    );


    it(
      'should traverse graph forward in correct order',
      async () => {
        const processor = new GraphProcessor(model.getGraph().getAllNodes(), GraphProcessorDirection.Forward);
        const callOrder: string[] = [];

        await processor.process(
          async (node: GraphNode): Promise<void> => {
            callOrder.push(node.getName());
          },
        );

        callOrder[0].should.equal('layer-1');
        callOrder[1].should.equal('layer-2');
        callOrder[2].should.equal('layer-3');
      },
    );


    it(
      'should traverse graph backward in correct order',
      async () => {
        const processor = new GraphProcessor(model.getGraph().getAllNodes(), GraphProcessorDirection.Backward);
        const callOrder: string[] = [];

        await processor.process(
          async (node: GraphNode): Promise<void> => {
            callOrder.push(node.getName());
          },
        );

        callOrder[0].should.equal('layer-3');
        callOrder[1].should.equal('layer-2');
        callOrder[2].should.equal('layer-1');
      },
    );
  },
);
