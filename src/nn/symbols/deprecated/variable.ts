import { NDSymbol } from './nd-symbol';


export class Variable extends NDSymbol {
  private frozen: boolean = false;


  public canOptimize(): boolean {
    return !this.frozen;
  }


  public freeze(): void {
    this.frozen = true;
  }
}

