import _ from 'lodash';
import { DeferredValueType, DeferredValue } from './value';
import { DeferredCollection } from './collection';


export class DeferredReadonlyCollection {
  private collection: DeferredCollection = new DeferredCollection();

  private requiredFields: string[] = [];


  public constructor(collection?: DeferredCollection|DeferredReadonlyCollection) {
    if (collection) {
      this.setCollection(collection);
    }
  }


  public get(key: string): DeferredValue {
    return this.collection.get(key);
  }


  public getValue(key: string): DeferredValueType {
    return this.collection.getValue(key);
  }


  public getDefault(): DeferredValue {
    return this.collection.getDefault();
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


  public setCollection(collection: DeferredCollection|DeferredReadonlyCollection): void {
    if (collection instanceof DeferredReadonlyCollection) {
      this.collection = (collection as DeferredReadonlyCollection).collection;
    }

    if (collection instanceof DeferredCollection) {
      this.collection = collection;
    }
  }


  public getRequiredFields(): string[] {
    return this.requiredFields;
  }
}
