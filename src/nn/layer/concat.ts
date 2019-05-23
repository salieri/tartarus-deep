import _ from 'lodash';
import { Layer, LayerParams } from './layer';
import { JoiEx, JoiExSchema } from '../../util';
import { DeferredValue, DeferredReadonlyCollection } from '../symbols';
import { NDArray } from '../../math';


export type ConcatOutputTraverseFunction = (field: DeferredValue, fieldKey: string, layerOutput: DeferredReadonlyCollection) => void;


export interface ConcatLayerExtendedDefinition {
  layer: string;
  field?: string|null;
}

export type ConcatLayerDefinition = string|ConcatLayerExtendedDefinition;


export interface ConcatParams extends LayerParams {
  order?: null|ConcatLayerDefinition[];
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
    if (this.params.order) {
      return _.map(
        this.params.order,
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
    return this.params.order ? this.params.order : this.rawInputs.getKeys();
  }


  protected verifyInputLayers(): void {
    const allKeys = this.rawInputs.getKeys();
    const orderedKeys = this.getInputKeysInOrder();

    const differenceAllKeys = _.difference(allKeys, orderedKeys);
    const differenceOrderedKeys = _.difference(orderedKeys, allKeys);

    if (differenceAllKeys.length > 0) {
      throw new Error(
        `Concat layer ${this.getName()} has more input layers than defined in the 'order' parameter: ${_.join(differenceAllKeys)}`,
      );
    }

    if (differenceOrderedKeys.length > 0) {
      throw new Error(
        `Concat layer ${this.getName()} 'order' parameter defines layers which output is `
        + `not linked to the layer: ${_.join(differenceOrderedKeys)}`,
      );
    }
  }


  protected traverse(callback: ConcatOutputTraverseFunction) : void {
    _.each(
      this.getInputInOrder(),
      (layer: ConcatLayerDefinition) => {
        const key = _.isString(layer) ? layer : layer.layer;
        const layerOutput = this.rawInputs.get(key);
        const fieldKey = _.isString(layer) ? layerOutput.getDefaultKey() : (layer.field || layerOutput.getDefaultKey());
        const field = layerOutput.get(fieldKey);

        callback(field, fieldKey, layerOutput);
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
        order: JoiEx.array()
          .optional()
          .items(
            JoiEx.string(),
            JoiEx.object().keys(
              {
                layer: JoiEx.string().required().description('Name of the layer'),
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
