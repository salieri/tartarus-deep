import { Model } from '../model';
import { Initializer, initializer, Layer } from '../../index';
import { RNNLayer, SoftmaxLayer } from '../../../index';


export interface BRNNModelFactoryOpts extends ModelFactoryOpts {
  depth           : number;
  c0Initializer?  : Initializer,
  a0Initializer?  : Initializer,
  rnnLayer?       : RNNLayer,
  outputLayer?    : Layer
}


export class BRNNFactory extends ModelFactory {
  public static factory(opts: BRNNModelFactoryOpts): Model {
    if (opts.depth < 2) {
      throw new Error('Depth must be at least 2');
    }

    const model     = new Model();
    const rnnLayer  = opts.rnnLayer;
    const c0Init    = opts.c0Initializer || new initializer.Zero();
    const a0Init    = opts.a0Initializer || new initializer.Zero();
    const x         = new SerialInput(opts.depth);

    model.input('x', x);

    _.times(
      opts.depth,
      (i) => {
        const t = i + 1;

        const rightBoundLayer = rnnLayer.clone(`rightbound<${t}>`);
        const leftBoundLayer  = rnnLayer.clone(`leftbound<${t}>`);

        model.register(rightBoundLayer, leftBoundLayer);

        let leftPrevious: RNNLayer;
        let rightPrevious: RNNLayer;

        leftBoundLayer.input('x<t>', x.at(t));
        rightBoundLayer.input('x<t>', x.at(t));

        if (opts.outputLayer) {
          const outputLayer = new SoftmaxLayer(`output<${t}>`);

          model.register(outputLayer);

          /* Should be equivalent to:
           * const c = new CombiningLayer(`combine<${t}>`);
           * c.input('a1', leftBoundLayer.output('a<t>');
           * c.input('a2', rightBoundLayer.output('a<t>');
           *
           * outputLayer.input('a', c.output('a'));
           */
          outputLayer.input('a', leftBoundLayer.output('a<t>'), rightBoundLayer.output('a<t>'));

          model.output(`yHat<${t}>`, outputLayer.output('c'));
        }

        if (i === 0) {
          // first entry
          leftBoundLayer.input('c<t-1>', c0Init.initialize(new NDArray(rnnLayer.getDimensions())));
          leftBoundLayer.input('a<t-1>', a0Init.initialize(new NDArray(rnnLayer.getDimensions())));

          rightBoundLayer.input('c<t-1>', c0Init.initialize(new NDArray(rnnLayer.getDimensions())));
          rightBoundLayer.input('a<t-1>', a0Init.initialize(new NDArray(rnnLayer.getDimensions())));
        } else {
          leftBoundLayer.input('c<t-1>', leftPrevious.output('c<t>'));
          leftBoundLayer.input('a<t-1>', leftPrevious.output('a<t>'));

          rightBoundLayer.input('c<t-1>', rightPrevious.output('c<t>'));
          rightBoundLayer.input('a<t-1>', rightPrevious.output('a<t>'));
        }

        leftPrevious  = leftBoundLayer;
        rightPrevious = rightBoundLayer;
      },
    );

    return model;
  }
}


