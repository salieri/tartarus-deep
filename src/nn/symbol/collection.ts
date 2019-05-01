import _ from 'lodash';
import { NDSymbol } from './nd-symbol';

export interface SymbolDescriptor {
  [key: string]: NDSymbol;
}


export class SymbolCollection {
  private symbols: SymbolDescriptor = {};

  public constructor(collection?: SymbolCollection) {
    if (collection) {
      this.symbols = _.cloneDeep(collection.symbols);
    }
  }


  public clear(): void {
    this.symbols = {};
  }


  public add(key: string, value: NDSymbol): void {
    if (this.has(key)) {
      throw new Error(`Key '${key} already exists in this collection`);
    }

    this.symbols[key] = value;
  }


  public set(key: string, value: NDSymbol): void {
    this.mustHave(key);

    this.symbols[key] = value;
  }


  public has(key: string): boolean {
    return (key in this.symbols);
  }


  protected mustHave(key: string): void {
    if (!this.has(key)) {
      throw new Error(`Unknown key '${key}'`);
    }
  }


  public get(key: string): NDSymbol {
    this.mustHave(key);

    return this.symbols[key];
  }


  public remove(key: string): void {
    delete this.symbols[key];
  }


  public clone(): SymbolCollection {
    return new SymbolCollection(this);
  }
}

