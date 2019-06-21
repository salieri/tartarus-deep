import _ from 'lodash';

import { DenseSimple2x } from '../../examples';
import { Model } from '../../src';
import { GraphNode } from '../../src/nn/graph';


describe(
  'Dense Network Lifecycle',
  () => {
    const generator = new DenseSimple2x();

    let model: Model;


    before(
      async () => {
        model = generator.model();

        await model.compile();
        await model.initialize();
      },
    );


    it(
      'should predict a result on input data',
      async () => {
        const result = await model.predict(4);
        const result2 = await model.predict(2);

        const nd = result.getDefaultValue();
        const nd2 = result2.getDefaultValue();

        nd.countDims().should.equal(1);
        nd2.countDims().should.equal(1);

        nd.countElements().should.equal(1);
        nd.countElements().should.equal(1);

        nd.getId().should.not.equal(nd2.getId());
        nd.equals(nd2).should.not.equal(true);
      },
    );


    it(
      'should evaluate model performance',
      async () => {
        const result = await model.evaluate(4, 8);

        const nd = result.losses.getDefaultValue();

        nd.countDims().should.equal(1);
        nd.countElements().should.equal(1);
      },
    );


    it.only(
      'should test the new fitter',
      async () => {
        const epochs = 20;
        const sampleCount = 20;
        const samples = generator.samples(sampleCount);

        await model.fitBetter(
          {
            epochs,
          },
          samples,
        );
      },
    );


    it(
      'should train the model with 100 samples',
      async () => {
        // const data = generator.samples(100);
        for (let i = 0; i < 10; i += 1) {
          const r = 3; // model.getSession().getRandomizer().intBetween(0, 10000);

          const prediction = (await model.predict(r)).getDefaultValue().sum();


console.log(`######################################## Round ${i} ########################################`);

console.log(`Expected ${r * 2}`);
console.log(`Predict: ${prediction}`);
console.log('');

_.each(
  model.getGraph().getAllNodes(),
  (node: GraphNode) => {
          console.log(
`${node.getName()}
bias: ${node.getEntity().data.optimizer.getValue('bias').data}
weight: ${node.getEntity().data.optimizer.getValue('weight').data}
`);
  }
);


          const result = await model.fit(r, r * 2);

          const loss = model.evaluation.losses.getDefaultValue().sum();
          const a = 1;
        }
      },
    );
  },
);
