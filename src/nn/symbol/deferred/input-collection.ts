import _ from 'lodash';
import { DeferredCollection } from './collection';
import { DeferredReadonlyCollection } from './readonly-collection';


export interface DeferredInputDictionary {
  [key: string]: DeferredReadonlyCollection;
}

export class DeferredInputCollection {
  public static readonly DEFAULT_INPUT: string = '__default__';

  private inputs: DeferredInputDictionary = {};


  public constructor(defaultInput?: DeferredReadonlyCollection|DeferredCollection) {
    if (defaultInput) {
      this.setDefault(defaultInput);
    }
  }


  public get(key: string): DeferredReadonlyCollection {
    if (!(key in this.inputs)) {
      throw new Error(`Key '${key}' not found`);
    }

    return this.inputs[key];
  }


  public set(key: string, input: DeferredReadonlyCollection|DeferredCollection): void {
    this.inputs[key] = new DeferredReadonlyCollection(input);
  }


  public setDefault(input: DeferredReadonlyCollection|DeferredCollection): void {
    this.inputs[DeferredInputCollection.DEFAULT_INPUT] = new DeferredReadonlyCollection(input);
  }


  public getDefault(): DeferredReadonlyCollection {
    return this.get(DeferredInputCollection.DEFAULT_INPUT);
  }


  public count(): number {
    return _.keys(this.inputs).length;
  }


  public first(): DeferredReadonlyCollection {
    const keys = _.keys(this.inputs);

    if (keys.length === 0) {
      throw new Error('No inputs available');
    }

    return this.inputs[keys[0]];
  }


  public merge(input: DeferredInputCollection, defaultFieldNameOverride: string|null = null, force: boolean = false): void {
    _.each(
      input.inputs,
      (sourceInput: DeferredReadonlyCollection, key: string) => {
        if ((!force) && (key in this.inputs)) {
          throw new Error(`Duplicate key: '${key}'`);
        }

        let finalKey = key;

        if ((finalKey === DeferredInputCollection.DEFAULT_INPUT) && (defaultFieldNameOverride)) {
          finalKey = defaultFieldNameOverride;
        }

        this.set(finalKey, sourceInput);
      },
    );
  }
}

