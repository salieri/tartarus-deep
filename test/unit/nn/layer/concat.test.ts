import {
  Concat,
} from '../../../../src';

import { LayerOutputUtil } from '../../../util';


describe(
  'Layer: Concat',
  () => {
    it(
      'should concatenate the output from its input layers',
      async () => {
        const concat = new Concat();

        const data = LayerOutputUtil.createOutput();

        concat.setRawInputs(data);

        await concat.compile();
        await concat.initialize();
        await concat.forward();

        const result = concat.output.getDefault();

        const expectedSize = data.get('layer-1').getDefault().countElements()
          + data.get('layer-2').getDefault().countElements();

        result.countElements().should.equal(expectedSize);

        const nd = result.get();

        nd.countDims().should.equal(1);
      },
    );


    it(
      'should allow specific output fields to be defined',
      async () => {
        const concat = new Concat({ fields: ['layer-1.secondField', 'layer-2'] });

        const data = LayerOutputUtil.createOutput();

        concat.setRawInputs(data);

        await concat.compile();
        await concat.initialize();
        await concat.forward();

        const result = concat.output.getDefault();

        const expectedSize = data.get('layer-1').get('secondField').countElements()
          + data.get('layer-2').getDefault().countElements();

        result.countElements().should.equal(expectedSize);

        const nd = result.get();

        nd.countDims().should.equal(1);
      },
    );


    it(
      'should use default output fields, if not told otherwise',
      async () => {
        const concat = new Concat({ fields: ['layer-1', 'layer-2'] });
        const data = LayerOutputUtil.createOutput();

        concat.setRawInputs(data);

        await concat.compile();
        await concat.initialize();
        await concat.forward();

        const result = concat.output.getDefault();

        const expectedSize = data.get('layer-1').getDefault().countElements()
          + data.get('layer-2').getDefault().countElements();

        result.countElements().should.equal(expectedSize);

        const nd = result.get();

        nd.countDims().should.equal(1);
      },
    );


    it(
      'should concatenate fields in specific order',
      async () => {
        const concat = new Concat({ fields: ['layer-2', 'layer-1'] });

        const data = LayerOutputUtil.createOutput();

        concat.setRawInputs(data);

        await concat.compile();
        await concat.initialize();
        await concat.forward();

        const result = concat.output.getDefault();

        const expectedSize = data.get('layer-1').getDefault().countElements()
          + data.get('layer-2').getDefault().countElements();

        result.countElements().should.equal(expectedSize);

        const nd = result.get();

        nd.countDims().should.equal(1);
        nd.countElements().should.equal(expectedSize);

        nd.getAt([0]).should.equal(5);
        nd.getAt([5]).should.equal(1);
        nd.getAt([6]).should.equal(1);
        nd.getAt([7]).should.equal(1);
        nd.getAt([8]).should.equal(1);
      },
    );


    it(
      'allow multiple fields from the same layer to be specified',
      async () => {
        const concat = new Concat(
          {
            fields: [
              { layer: 'layer-1', field: 'secondField' },
              'layer-1.firstField',
              'layer-2',
            ],
          },
        );

        const data = LayerOutputUtil.createOutput();

        concat.setRawInputs(data);

        await concat.compile();
        await concat.initialize();
        await concat.forward();

        const result = concat.output.getDefault();

        const expectedSize = data.get('layer-1').get('secondField').countElements()
          + data.get('layer-1').get('firstField').countElements()
          + data.get('layer-2').getDefault().countElements();

        result.countElements().should.equal(expectedSize);

        const nd = result.get();

        nd.countDims().should.equal(1);
        nd.countElements().should.equal(expectedSize);

        nd.getAt([0]).should.equal(3);
        nd.getAt([1]).should.equal(3);
        nd.getAt([2]).should.equal(3);

        nd.getAt([3]).should.equal(1);
        nd.getAt([3 + 4]).should.equal(2);
        nd.getAt([3 + 4 + 4]).should.equal(5);
        nd.getAt([3 + 4 + 4 + 4]).should.equal(5);
      },
    );


    it(
      'should fail, if field configuration refers to a non-existent field',
      async () => {
        const concat = new Concat({ fields: ['layer-1.nonExistentField', 'layer-2'] });
        const data = LayerOutputUtil.createOutput();

        concat.setRawInputs(data);

        await concat.compile().should.be.rejectedWith(/Concat layer .* requires field .* from layer .* which has not been declared/);
      },
    );


    it(
      'should fail, if field configuration refers to a non-existent layer',
      async () => {
        const concat = new Concat({ fields: ['missing-layer'] });
        const data = LayerOutputUtil.createOutput();

        concat.setRawInputs(data);

        await concat.compile().should.be.rejectedWith(/Concat layer .* expects input from layer .* which is not linked to this layer/);
      },
    );


    it(
      'should fail, if not all input layers are referred',
      async () => {
        const concat = new Concat({ fields: ['layer-1'] });
        const data = LayerOutputUtil.createOutput();

        concat.setRawInputs(data);

        await concat.compile().should.be.rejectedWith(/Concat layer .* has more input layers than defined in the .* parameter/);
      },
    );
  },
);
