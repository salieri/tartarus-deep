import _ from 'lodash';
import { DeferredCollection } from './collection';
import { DeferredReadonlyCollection } from './readonly-collection';
import { KeyNotFoundError } from '../../../error';


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
      throw new KeyNotFoundError(`Key '${key}' not found`, key);
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


  public getKeys(): string[] {
    return _.keys(this.inputs);
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


  public merge(input: DeferredInputCollection, defaultFieldNameRemap: string|null = null, force: boolean = false): void {
    _.each(
      input.inputs,
      (sourceInput: DeferredReadonlyCollection, key: string) => {
        if ((!force) && (key in this.inputs)) {
          throw new Error(`Duplicate key: '${key}'`);
        }

        let finalKey = key;

        if ((finalKey === DeferredInputCollection.DEFAULT_INPUT) && (defaultFieldNameRemap)) {
          finalKey = defaultFieldNameRemap;
        }

        this.set(finalKey, sourceInput);
      },
    );
  }


  public has(key: string): boolean {
    return (key in this.inputs);
  }


  public assign(inputs: DeferredInputCollection): void {
    _.each(
      inputs.getKeys(),
      (key: string) => {
        if (!this.has(key)) {
          throw new Error(`Cannot assign an undeclared input: ${key}`);
        }

        const collection = this.get(key);

        collection.assign(inputs.get(key));
      },
    );
  }


  public areAllSet(): boolean {
    return _.every(
      this.inputs,
      (input: DeferredReadonlyCollection) => input.areAllSet(),
    );
  }


  public areAllDeclared(): boolean {
    return _.every(
      this.inputs,
      (input: DeferredReadonlyCollection) => input.areAllDeclared(),
    );
  }


  public unsetValues(): void {
    _.each(
      this.inputs,
      (input: DeferredReadonlyCollection) => input.getCollection().unsetValues(),
    );
  }


  /* public filter(matchList: string[], convertSingleToDefault: boolean = false): DeferredInputCollection {
    if (matchList.length === 0) {
      throw new Error('Empty match list');
    }

    if ((matchList.length === 1) && (convertSingleToDefault)) {
      return new DeferredInputCollection(this.first());
    }

    const collection = new DeferredInputCollection();

    _.each(
      _.intersection(_.keys(this.inputs), matchList),
      (key: string) => (collection.set(key, this.get(key))),
    );

    return collection;
  } */
}

