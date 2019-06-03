import { NDArray } from '../../src/math';
import { DeferredInputCollection, DeferredCollection } from '../../src/nn/symbols';
import { Layer } from '../../src/nn';

export class LayerOutputUtil {
  public static createManyOutputs(): DeferredInputCollection {
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


  public static createOutput(): DeferredInputCollection {
    const output = new DeferredCollection();

    output.declare('values', 4);
    output.setValue('values', new NDArray([4, 5, 6, 7]));

    output.setDefaultKey('values');

    const layerOutput = new DeferredInputCollection();

    layerOutput.setDefault(output);

    return layerOutput;
  }


  public static createBackpropOutput(): DeferredInputCollection {
    const output = new DeferredCollection();

    output.declare(Layer.DERIVATIVE, 4);
    output.setValue(Layer.DERIVATIVE, new NDArray([0.4, 0.3, 0.2, 0.1]));

    output.declare(Layer.LOSS, 1);
    output.setValue(Layer.LOSS, new NDArray([0.8]));

    const layerBackpropOutput = new DeferredInputCollection();

    layerBackpropOutput.setDefault(output);

    return layerBackpropOutput;
  }
}
