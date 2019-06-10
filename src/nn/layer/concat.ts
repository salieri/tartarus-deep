import _ from 'lodash';
import { Layer, LayerParams } from './layer';
import { JoiEx, JoiExSchema } from '../../util';

import {
  DeferredValue,
  DeferredCollectionWrapper,
  DeferredInputCollection,
  DeferredCollection,
} from '../symbols';

import { NDArray } from '../../math';
import { KeyNotFoundError } from '../../error';


export type ConcatOutputTraverseFunction =
  (field: DeferredValue, fieldKey: string, layerOutput: DeferredCollectionWrapper, layerKey: string) => void;

export type ConcatOutputTraverseKeyFunction = (fieldKey: string, layerOutput: DeferredCollectionWrapper, layerKey: string) => void;


export interface ConcatLayerExtendedDefinition {
  layer: string;
  field?: string|null;
}

export type ConcatLayerDefinition = string|ConcatLayerExtendedDefinition;


export interface ConcatParams extends LayerParams {
  fields?: null|ConcatLayerDefinition[];
}


export class Concat extends Layer<ConcatParams> {
  public static readonly CONCATENATED: string = 'concatenated';

  protected rawBackpropOutputs = new DeferredInputCollection();


  public async backwardExec(): Promise<void> {
    const nd = this.backpropInput.getValue(Layer.DERIVATIVE);
    const loss = this.backpropInput.getValue(Layer.LOSS);

    let curPos = 0;

    this.traverse(
      (field: DeferredValue, fieldKey: string, layerOutput: DeferredCollectionWrapper, layerKey: string) => {
        if (layerOutput.getDefaultKey() !== fieldKey) {
          curPos += field.countElements();
          return; // only default output will have a derivative
        }

        const coll = this.rawBackpropOutputs.get(layerKey).getCollection();

        const derivative = nd.slice([curPos], coll.get(Layer.DERIVATIVE).countElements());

        coll.setValue(Layer.LOSS, loss);
        coll.setValue(Layer.DERIVATIVE, derivative);

        curPos += field.countElements();
      },
    );
  }


  public async forwardExec(): Promise<void> {
    let result: NDArray|undefined;

    this.traverse(
      (field: DeferredValue): void => {
        const fieldValue = field.get().flatten();

        result = result ? result.concat(fieldValue) : fieldValue;
      },
    );

    if (!result) {
      throw new Error('No layers to concatenate');
    }

    this.output.setDefaultValue(result);
  }


  protected getInputInOrder(): ConcatLayerDefinition[] {
    return this.params.fields ? this.params.fields : this.rawInputs.getKeys();
  }


  protected verifyInputLayers(): void {
    const allKeys = this.rawInputs.getKeys();
    // const orderedKeys = this.getInputKeysInOrder();

    const definedLayerKeys: string[] = [];

    try {
      this.traverseKeys(
        (fieldKey: string, layerOutput: DeferredCollectionWrapper, layerKey: string) => {
          definedLayerKeys.push(layerKey);

          try {
            layerOutput.require(fieldKey);
          } catch (err) {
            if (err instanceof KeyNotFoundError) {
              throw new Error(`Concat layer '${this.getName()}' requires `
                + `field '${fieldKey}' from `
                + `layer '${layerKey}', which has not been declared`);
            }

            throw err;
          }
        },
      );
    } catch (err) {
      if (err instanceof KeyNotFoundError) {
        throw new Error(`Concat layer '${this.getName()}' expects input from layer '${err.key}', which is not linked to this layer`);
      }

      throw err;
    }

    const cleanedLayerKeys = _.uniq(definedLayerKeys);
    const differenceAllKeys = _.difference(allKeys, cleanedLayerKeys);
    // const differenceOrderedKeys = _.difference(cleanedLayerKeys, allKeys);

    if (differenceAllKeys.length > 0) {
      throw new Error(
        `Concat layer '${this.getName()}' has more input layers than defined in the 'fields' parameter: ${_.join(differenceAllKeys)}`,
      );
    }

    // if (differenceOrderedKeys.length > 0) {
    //   throw new Error(
    //     `Concat layer ${this.getName()} 'fields' parameter defines layers which output is `
    //     + `not linked to the layer: ${_.join(differenceOrderedKeys)}`,
    //   );
    // }
  }


  protected traverse(callback: ConcatOutputTraverseFunction) : void {
    this.traverseKeys(
      (fieldKey: string, layerOutput: DeferredCollectionWrapper, layerKey: string) => {
        const field = layerOutput.get(fieldKey);

        callback(field, fieldKey, layerOutput, layerKey);
      },
    );
  }


  protected traverseKeys(callback: ConcatOutputTraverseKeyFunction) : void {
    _.each(
      this.getInputInOrder(),
      (layer: ConcatLayerDefinition) => {
        let layerKey = _.isString(layer) ? layer : layer.layer;

        let fieldKey: string|undefined;

        const layerSections = _.split(layerKey, '.', 2);

        if (layerSections.length > 1) {
          layerKey = layerSections[0];
          fieldKey = layerSections[1];
        }

        const layerOutput = this.rawInputs.get(layerKey);

        if (!fieldKey) {
          fieldKey = _.isString(layer) ? layerOutput.getDefaultKey() : (layer.field || layerOutput.getDefaultKey());
        }

        callback(fieldKey, layerOutput, layerKey);
      },
    );
  }


  protected determineOutputSize(): number {
    let total = 0;

    this.traverse(
      (field: DeferredValue, fieldKey: string, layerOutput: DeferredCollectionWrapper): void => {
        layerOutput.require(fieldKey);

        total += field.countElements();
      },
    );

    return total;
  }


  public async compileForwardPropagation(): Promise<void> {
    this.verifyInputLayers();

    this.output.declare(Concat.CONCATENATED, this.determineOutputSize());
    this.output.setDefaultKey(Concat.CONCATENATED);

    this.prepareForBackprop();
  }


  protected prepareForBackprop(): void {
    const layers:string[] = [];

    this.traverseKeys((fieldKey: string, layerOutput: DeferredCollectionWrapper, layerKey: string) => layers.push(layerKey));

    _.each(_.uniq(layers), (layer: string) => this.rawBackpropOutputs.set(layer, new DeferredCollection()));
  }


  public async compileBackPropagation(): Promise<void> {
    this.traverse(
      (field: DeferredValue, fieldKey: string, layerOutput: DeferredCollectionWrapper, layerKey: string): void => {
        if (layerOutput.getDefaultKey() !== fieldKey) {
          return; // Only default output will have a derivative
        }

        const bpOutput = this.rawBackpropOutputs.get(layerKey).getCollection();

        if (!bpOutput.has(Layer.LOSS)) {
          bpOutput.declare(Layer.LOSS, 1);
        }

        if (!bpOutput.has(Layer.DERIVATIVE)) {
          bpOutput.declare(Layer.DERIVATIVE, field.getDims());
        }
      },
    );
  }


  public getRawBackpropOutputs(): DeferredInputCollection {
    return this.rawBackpropOutputs;
  }


  public unsetBackpropOutputValues(): void {
    this.rawBackpropOutputs.unsetValues();
  }


  public async initializeExec(): Promise<void> {
    // do nothing
  }


  public getParamSchema(): JoiExSchema {
    return JoiEx.object().keys(
      {
        fields: JoiEx.array()
          .optional()
          .items(
            JoiEx.string(),
            JoiEx.object().keys(
              {
                layer: JoiEx.string().required().description('Name of the layer ("layer.field" shortcut allowed)'),
                field: JoiEx.string().optional().allow(null).default(null)
                  .description('Output field to include'),
              },
            ),
          )
          .allow(null)
          .default(null)
          .description('Order in which layers are concatenated'),
      },
    );
  }
}
