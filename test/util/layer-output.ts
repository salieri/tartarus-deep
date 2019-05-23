import { NDArray } from '../../src/math';
import { DeferredInputCollection, DeferredCollection } from '../../src/nn/symbols';

export class LayerOutputUtil {
  public static createOutput(): DeferredInputCollection {
    const layer1Output = new DeferredCollection();

    layer1Output.declare('firstField', [2, 4]);
    layer1Output.declare('secondField', [3]);

    layer1Output.setValue('firstField', new NDArray([[1, 1, 1, 1], [2, 2, 2, 2]]));
    layer1Output.setValue('secondField', new NDArray([3, 3, 3]));

    layer1Output.setDefaultKey('firstField');


    const layer2Output = new DeferredCollection();

    layer2Output.declareDefault(5);

    layer2Output.setDefaultValue(new NDArray([5, 5, 5, 5, 5]));


    const layerOutput = new DeferredInputCollection();

    layerOutput.set('layer-1', layer1Output);
    layerOutput.set('layer-2', layer2Output);

    return layerOutput;
  }
}
