import _ from 'lodash';
import { DeferredCollection } from './collection';
import { DeferredCollectionWrapper } from './collection-wrapper';
import { KeyNotFoundError } from '../../../error';
import { Matrix, NDArray, Vector } from '../../../math';

export type KeyReassignCallback = (key: string) => string;

export interface DeferredInputDictionary {
  [key: string]: DeferredCollectionWrapper;
}

export class DeferredInputCollection {
  private static idCounter: number = 0;

  public static readonly DEFAULT_INPUT: string = '__default__';

  private inputs: DeferredInputDictionary = {};

  private id: number;


  public constructor(defaultInput?: DeferredCollectionWrapper|DeferredCollection) {
    if (defaultInput) {
      this.setDefault(defaultInput);
    }

    DeferredInputCollection.idCounter += 1;

    this.id = DeferredInputCollection.idCounter;
  }


  public get(key: string): DeferredCollectionWrapper {
    if (!(key in this.inputs)) {
      throw new KeyNotFoundError(`Key '${key}' not found`, key);
    }

    return this.inputs[key];
  }


  public set(key: string, input: DeferredCollectionWrapper|DeferredCollection): void {
    this.inputs[key] = new DeferredCollectionWrapper(input);
  }


  public setDefault(input: DeferredCollectionWrapper|DeferredCollection): void {
    this.inputs[DeferredInputCollection.DEFAULT_INPUT] = new DeferredCollectionWrapper(input);
  }


  public getDefault(): DeferredCollectionWrapper {
    return this.get(DeferredInputCollection.DEFAULT_INPUT);
  }


  public getKeys(): string[] {
    return _.keys(this.inputs);
  }


  public count(): number {
    return _.keys(this.inputs).length;
  }


  public first(): DeferredCollectionWrapper {
    const keys = _.keys(this.inputs);

    if (keys.length === 0) {
      throw new Error('No inputs available');
    }

    return this.inputs[keys[0]];
  }


  public merge(
    input: DeferredInputCollection,
    defaultFieldNameRemap: string|null = null,
    force: boolean = false,
    keyReassignCb?: KeyReassignCallback,
  ): void {
    _.each(
      input.inputs,
      (sourceInput: DeferredCollectionWrapper, key: string) => {
        let finalKey = key;

        if ((finalKey === DeferredInputCollection.DEFAULT_INPUT) && (defaultFieldNameRemap)) {
          finalKey = defaultFieldNameRemap;
        }

        if (keyReassignCb) {
          finalKey = keyReassignCb(finalKey);
        }

        if ((!force) && (finalKey in this.inputs)) {
          throw new Error(`Duplicate key: '${finalKey}'`);
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
      (input: DeferredCollectionWrapper) => input.areAllSet(),
    );
  }


  public areAllDeclared(): boolean {
    return _.every(
      this.inputs,
      (input: DeferredCollectionWrapper) => input.areAllDeclared(),
    );
  }


  public unsetValues(): void {
    _.each(
      this.inputs,
      (input: DeferredCollectionWrapper) => input.getCollection().unsetValues(),
    );
  }


  public filter(keys: string[], allowMissing: boolean = false): DeferredInputCollection {
    const result = new DeferredInputCollection();

    _.each(
      keys,
      (key: string) => {
        if (!this.has(key)) {
          if (!allowMissing) {
            throw new KeyNotFoundError(`Missing key: ${key}`, key);
          }

          return;
        }

        result.set(key, this.get(key));
      },
    );

    return result;
  }


  public getDefaultValue<NDType extends NDArray = NDArray>(type?: { new (val: NDArray|Matrix|Vector): NDType }): NDType {
    return this.getDefault().getDefault().get(type);
  }


  /**
  * Returns an UNCONNECTED snapshot
  */
  public snapshot(): DeferredInputCollection {
    if (!this.areAllSet()) {
      throw new Error('Cannot snapshopt -- some declared values in the input collection are not set');
    }

    return _.cloneDeep(this);
  }


  /**
   * Returns a CONNECTED clone
   */
  public clone(): DeferredInputCollection {
    const c = new DeferredInputCollection();

    _.each(this.getKeys(), (k: string) => c.set(k, this.get(k)));

    return c;
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

