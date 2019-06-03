import _ from 'lodash';
import { DeferredValueType, DeferredValue } from './value';
import { DeferredCollection } from './collection';


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
