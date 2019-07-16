import { Dense } from '../../../../src/nn/layer';
import { Model } from '../../../../src/nn/model';
import { NDArray } from '../../../../src/math';


describe(
  'Graph',
  () => {
    const model = new Model();
    const initialInput = new NDArray([0, 0.1]);

    const d1 = new Dense({ units: 3 }, 'layer-1');
    const d2 = new Dense({ units: 4 }, 'layer-2');
    const d3 = new Dense({ units: 2 }, 'layer-3');

    before(
      async () => {
        model
          .input(2)
          .push(d1)
          .push(d2)
          .push(d3);

        await model.compile();

        model.getGraph().assignInput(Model.coerceData(initialInput));
      },
    );


    it(
      'should find layers by index',
      () => {
        model.getGraph().find(0).getEntity().should.equal(d1);
        model.getGraph().find(1).getEntity().should.equal(d2);
        model.getGraph().find(2).getEntity().should.equal(d3);
      },
    );


    it(
      'should find layers by entity',
      () => {
        const g = model.getGraph();

        g.find(d1).should.equal(g.find(0));
        g.find(d2).should.equal(g.find(1));
        g.find(d3).should.equal(g.find(2));
      },
    );

    it(
      'should find layers by graph node',
      () => {
        const g = model.getGraph();

        const n = g.find(d2);

        g.find(n).should.equal(n);
      },
    );


    it(
      'should prevent modifications after compilation',
      async () => {
        const m2 = new Model();

        // @ts-ignore
        m2.getGraph().canModify();

        m2.input(1);
        m2.push(new Dense({ units: 3 }));

        await m2.compile();

        // @ts-ignore
        (() => m2.getGraph().canModify()).should.Throw(/Graph cannot be modified after compilation/);
      },
    );
  },
);
