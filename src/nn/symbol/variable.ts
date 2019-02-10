import { Symbol } from './symbol';


export class Variable extends Symbol {
  private frozen: boolean = false;


  public canOptimize(): boolean {
    return !this.frozen;
  }


  public freeze(): void {
    this.frozen = true;
  }
}

