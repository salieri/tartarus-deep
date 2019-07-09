import _ from 'lodash';
import { Layer, LayerParams } from './layer';
import { JoiEx, JoiExSchema } from '../../util';

import {
  DeferredValue,
  DeferredCollectionWrapper,
  DeferredCollection,
} from '../symbols';

import { Matrix, MatrixDirection, NDArray, Vector } from '../../math';
import { KeyNotFoundError } from '../../error';
import { Dense } from './dense';


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


  protected async optimizeExec(): Promise<void> {
    // do nothing
  }


  protected resolveBackpropInput(): void {
    const backpropInput = this.data.backpropInput;
    const rawBackpropInputs = this.raw.backpropInputs;

    // This needs rewriting to deal with cases where a layer has multiple outputs
    // This needs rewriting to deal with bias, not just weight
    try {
      backpropInput.setCollection(rawBackpropInputs.getDefault());
      return;
    } catch (err) {
      if (!(err instanceof KeyNotFoundError)) {
        throw err;
      }
    }

    if (rawBackpropInputs.count() < 1) {
      throw new Error(`Missing backprop input for concat layer '${this.getName()}' -- concat layer cannot be an output layer`);
    }

    if (rawBackpropInputs.count() > 1) {
      // throw new Error(`Too many inputs for a dense layer '${this.getName()}'`);
      // DO SOMETHING
    }

    backpropInput.setCollection(rawBackpropInputs.first());
  }


  protected async backwardExec(): Promise<void> {
    const bpInput = this.data.backpropInput;

    const errorTerm = bpInput.getValue(Layer.ERROR_TERM, Vector);
    const w = bpInput.getValue(Dense.WEIGHT_MATRIX, Matrix);

    let curPos = 0;

    this.traverse(
      (field: DeferredValue, fieldKey: string, layerOutput: DeferredCollectionWrapper, layerKey: string) => {
        if (layerOutput.getDefaultKey() !== fieldKey) {
          curPos += field.countElements();
          return; // only default output will have a derivative
        }

        const wm = w.slice(MatrixDirection.Vertical, curPos, field.countElements());
        const coll = this.raw.backpropOutputs.get(layerKey).getCollection();

        coll.setValue(Dense.WEIGHT_MATRIX, wm);
        coll.setValue(Layer.ERROR_TERM, errorTerm);

        curPos += field.countElements();
      },
    );
  }


  protected async forwardExec(): Promise<void> {
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

    this.data.output.setDefaultValue(result);
  }


  protected getInputInOrder(): ConcatLayerDefinition[] {
    return this.params.fields ? this.params.fields : this.raw.inputs.getKeys();
  }


  protected verifyInputLayers(): void {
    const allKeys = this.raw.inputs.getKeys();
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

        const layerOutput = this.raw.inputs.get(layerKey);

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


  protected async compileInitialization(): Promise<void> {
    this.raw.outputs.setDefault(this.data.output);
  }


  protected async compileForwardPropagation(): Promise<void> {
    this.verifyInputLayers();

    this.data.output.declare(Concat.CONCATENATED, this.determineOutputSize());
    this.data.output.setDefaultKey(Concat.CONCATENATED);

    this.prepareForBackprop();
  }


  protected prepareForBackprop(): void {
    const layers:string[] = [];

    this.traverseKeys((fieldKey: string, layerOutput: DeferredCollectionWrapper, layerKey: string) => layers.push(layerKey));

    _.each(_.uniq(layers), (layer: string) => this.raw.backpropOutputs.set(layer, new DeferredCollection()));
  }


  protected async compileBackPropagation(): Promise<void> {
    this.resolveBackpropInput();

    const backpropInput = this.data.backpropInput;

    backpropInput.require(Layer.ERROR_TERM);
    backpropInput.require(Dense.WEIGHT_MATRIX);

    this.traverse(
      (field: DeferredValue, fieldKey: string, layerOutput: DeferredCollectionWrapper, layerKey: string): void => {
        if (layerOutput.getDefaultKey() !== fieldKey) {
          return; // Only default output will have a derivative
        }

        const bpOutput = this.raw.backpropOutputs.get(layerKey).getCollection();
        const errorTerm = backpropInput.get(Layer.ERROR_TERM);
        const nextLayerUnits = errorTerm.countElements();
        const inputUnits = this.raw.inputs.get(layerKey).get(fieldKey).countElements();

        bpOutput.declare(Dense.WEIGHT_MATRIX, [nextLayerUnits, inputUnits]);
        bpOutput.declare(Layer.ERROR_TERM, errorTerm.getDims());
      },
    );
  }


  protected async initializeExec(): Promise<void> {
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
