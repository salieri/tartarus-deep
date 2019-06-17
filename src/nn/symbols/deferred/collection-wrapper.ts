import _ from 'lodash';
import { DeferredValue } from './value';
import { DeferredCollection } from './collection';
import { Matrix, Vector, NDArray } from '../../../math';


export class DeferredCollectionWrapper {
  private static idCounter: number = 0;

  private collection: DeferredCollection = new DeferredCollection();

  private requiredFields: string[] = [];

  private id: number;


  public constructor(collection?: DeferredCollection|DeferredCollectionWrapper) {
    if (collection) {
      this.setCollection(collection);
    }

    DeferredCollectionWrapper.idCounter += 1;

    this.id = DeferredCollectionWrapper.idCounter;
  }


  public get(key: string): DeferredValue {
    return this.collection.get(key);
  }


  public getValue<NDType extends NDArray = NDArray>(key: string, type?: { new (val: NDArray|Matrix|Vector): NDType }): NDType {
    return this.collection.getValue(key, type);
  }


  public getDefault(): DeferredValue {
    return this.collection.getDefault();
  }


  public getDefaultValue<NDType extends NDArray = NDArray>(type?: { new (val: NDArray|Matrix|Vector): NDType }): NDType {
    return this.collection.getDefaultValue(type);
  }


  public has(key: string): boolean {
    return this.collection.has(key);
  }


  public hasDefaultValue(): boolean {
    return this.collection.has(this.collection.getDefaultKey());
  }


  public getDefaultKey(): string {
    return this.collection.getDefaultKey();
  }


  public require(key: string): void {
    this.collection.require(key);

    this.requiredFields.push(key);

    this.requiredFields = _.uniq(this.requiredFields);
  }


  public requireDefault(): void {
    this.require(this.collection.getDefaultKey());
  }


  public setCollection(collection: DeferredCollection|DeferredCollectionWrapper): void {
    if (collection instanceof DeferredCollectionWrapper) {
      this.collection = (collection as DeferredCollectionWrapper).collection;
    }

    if (collection instanceof DeferredCollection) {
      this.collection = collection;
    }
  }


  public getCollection(): DeferredCollection {
    return this.collection;
  }


  public areAllSet(): boolean {
    return this.collection.areAllSet();
  }


  public areAllDeclared(): boolean {
    return this.collection.areAllDeclared();
  }


  public getRequiredFields(): string[] {
    return this.requiredFields;
  }


  public assign(collection: DeferredCollectionWrapper): void {
    this.collection.assign(collection.collection);
  }


  public unsetValues(): void {
    this.collection.unsetValues();
  }
}
