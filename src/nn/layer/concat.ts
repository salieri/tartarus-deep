import _ from 'lodash';
import { Layer, LayerParams } from './layer';
import { JoiEx, JoiExSchema } from '../../util';
import { DeferredValue, DeferredReadonlyCollection } from '../symbols';
import { NDArray } from '../../math';
import { KeyNotFoundError } from '../../error';


export type ConcatOutputTraverseFunction = (field: DeferredValue, fieldKey: string, layerOutput: DeferredReadonlyCollection, layerKey: string) => void;
export type ConcatOutputTraverseKeyFunction = (fieldKey: string, layerOutput: DeferredReadonlyCollection, layerKey: string) => void;


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


  public async backwardExec(): Promise<void> {
    // nothing yet
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


  protected getInputKeysInOrder(): string[] {
    if (this.params.fields) {
      return _.map(
        this.params.fields,
        (l: ConcatLayerDefinition): string => {
          if (_.isString(l)) {
            return l;
          }

          return l.layer;
        },
      );
    }

    return this.rawInputs.getKeys();
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
        (fieldKey: string, layerOutput: DeferredReadonlyCollection, layerKey: string) => {
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
    const differenceOrderedKeys = _.difference(cleanedLayerKeys, allKeys);

    if (differenceAllKeys.length > 0) {
      throw new Error(
        `Concat layer '${this.getName()}' has more input layers than defined in the 'fields' parameter: ${_.join(differenceAllKeys)}`,
      );
    }

    if (differenceOrderedKeys.length > 0) {
      throw new Error(
        `Concat layer ${this.getName()} 'fields' parameter defines layers which output is `
        + `not linked to the layer: ${_.join(differenceOrderedKeys)}`,
      );
    }
  }


  protected traverse(callback: ConcatOutputTraverseFunction) : void {
    this.traverseKeys(
      (fieldKey: string, layerOutput: DeferredReadonlyCollection, layerKey: string) => {
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
      (field: DeferredValue, fieldKey: string, layerOutput: DeferredReadonlyCollection): void => {
        layerOutput.require(fieldKey);

        total += field.countElements();
      },
    );

    return total;
  }


  public async compileExec(): Promise<void> {
    this.verifyInputLayers();

    this.output.declare(Concat.CONCATENATED, this.determineOutputSize());
    this.output.setDefaultKey(Concat.CONCATENATED);
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
