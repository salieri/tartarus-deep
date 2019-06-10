import _ from 'lodash';

import { Concat, Dense, Model } from '../../../../src/nn';
import { GraphNode, GraphProcessor, GraphProcessorDirection } from '../../../../src/nn/graph';
import { KeyNotFoundError } from '../../../../src/error';

describe(
  'Node Processor',
  () => {
    const model = new Model();

    it(
      'should prepare a test model',
      async () => {
        model.input(2);

        const l1 = new Dense({ units: 3 }, 'layer-1');
        const l2a = new Dense({ units: 3 }, 'layer-2a');
        const l2b = new Dense({ units: 3 }, 'layer-2b');
        const l3 = new Concat({}, 'layer-3');

        model.add(l1);
        model.add(l2a, l1);
        model.add(l2b, l1);
        model.add(l3, [l2a, l2b]);

        model.output(l3);

        await model.compile();

        model.getGraph().assignInput(Model.coerceData([1, 2]));
      },
    );


    it(
      'should traverse graph forward in correct order',
      async () => {
        const processor = new GraphProcessor(model.getGraph(), GraphProcessorDirection.Forward);
        const callOrder: string[] = [];
        const paramName = 'test_forwardprop';

        await processor.process(
          async (node: GraphNode): Promise<void> => {
            callOrder.push(node.getName());

            node.devParams.set(paramName, true);
          },
          (node: GraphNode): boolean => {
            if (node.getName() === 'layer-1') {
              return true;
            }

            return _.every(
              node.getInputNodes(),
              (n: GraphNode) => {
                try {
                  return n.devParams.get(paramName) as boolean;
                } catch (err) {
                  if (err instanceof KeyNotFoundError) {
                    return false;
                  }

                  throw err;
                }
              },
            );
          },
        );

        callOrder[0].should.equal('layer-1');
        callOrder[1].should.equal('layer-2a');
        callOrder[2].should.equal('layer-2b');
        callOrder[3].should.equal('layer-3');
      },
    );


    it(
      'should traverse graph backward in correct order',
      async () => {
        const processor = new GraphProcessor(model.getGraph(), GraphProcessorDirection.Backward);
        const callOrder: string[] = [];
        const paramName = 'test_backprop';

        await processor.process(
          async (node: GraphNode): Promise<void> => {
            callOrder.push(node.getName());

            node.devParams.set(paramName, true);
          },
          (node: GraphNode): boolean => {
            if (node.getName() === 'layer-3') {
              return true;
            }

            return _.every(
              node.getOutputNodes(),
              (n: GraphNode) => {
                try {
                  return n.devParams.get(paramName) as boolean;
                } catch (err) {
                  if (err instanceof KeyNotFoundError) {
                    return false;
                  }

                  throw err;
                }
              },
            );
          },
        );

        callOrder[0].should.equal('layer-3');
        callOrder[1].should.equal('layer-2a');
        callOrder[2].should.equal('layer-2b');
        callOrder[3].should.equal('layer-1');
      },
    );
  },
);
