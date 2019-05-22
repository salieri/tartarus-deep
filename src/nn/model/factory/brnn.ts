
// import { Model } from '../model';
// import { Initializer, initializer, Layer } from '../../index';
// import { RNNLayer, SoftmaxLayer } from '../../../index';
//
//
// export interface BRNNModelFactoryOpts extends ModelFactoryOpts {
//   depth           : number;
//   rnnLayer        : RNNLayer;
//   c0Initializer?  : Initializer;
//   a0Initializer?  : Initializer;
//   outputLayer?    : Layer;
// }
//
//
export class BRNNFactory {
//   public static factory(opts: BRNNModelFactoryOpts): Model {
//     if (opts.depth < 2) {
//       throw new Error('Depth must be at least 2');
//     }
//
//     const model     = new Model();
//     const rnnLayer  = opts.rnnLayer;
//     const c0Init    = opts.c0Initializer || new initializer.Zero();
//     const a0Init    = opts.a0Initializer || new initializer.Zero();
//     const x         = new SerialInput(opts.depth, rnnLayer.getDimensions());
//     const outputs   = [];
//
//     model.input(x);
//
//     _.times(
//       opts.depth,
//       (i) => {
//         const t = i + 1;
//
//         const rightBoundLayer = rnnLayer.clone(`rightbound<${t}>`);
//         const leftBoundLayer  = rnnLayer.clone(`leftbound<${t}>`);
//
//         model.register(rightBoundLayer, leftBoundLayer);
//
//         let leftPrevious: RNNLayer;
//         let rightPrevious: RNNLayer;
//
//         leftBoundLayer.input.set(x.at(t));
//         rightBoundLayer.input.set(x.at(t));
//
//         if (opts.outputLayer) {
//           const outputLayer = new SoftmaxLayer(`output<${t}>`);
//
//           outputLayer.input.set(Merge(leftBoundLayer.output(), rightBoundLayer.output()));
//
//           model.register(outputLayer);
//
//           outputs.push(outputLayer.output());
//         }
//
//         if (i === 0) {
//           // first entry
//           leftBoundLayer.cache.set('c<t-1>', c0Init.initialize(new NDArray(rnnLayer.getDimensions())));
//           leftBoundLayer.cache.set('a<t-1>', a0Init.initialize(new NDArray(rnnLayer.getDimensions())));
//
//           rightBoundLayer.cache.set('c<t-1>', c0Init.initialize(new NDArray(rnnLayer.getDimensions())));
//           rightBoundLayer.cache.set('a<t-1>', a0Init.initialize(new NDArray(rnnLayer.getDimensions())));
//         } else {
//           leftBoundLayer.cache.set('c<t-1>', leftPrevious.cache.get('c<t>'));
//           leftBoundLayer.cache.set('a<t-1>', leftPrevious.cache.get('a<t>'));
//
//           rightBoundLayer.cache.set('c<t-1>', rightPrevious.cache.get('c<t>'));
//           rightBoundLayer.cache.set('a<t-1>', rightPrevious.cache.get('a<t>'));
//         }
//
//         leftPrevious  = leftBoundLayer;
//         rightPrevious = rightBoundLayer;
//       },
//     );
//
//     model.output.set(Eventually(outputs));
//
//     return model;
//   }
}


