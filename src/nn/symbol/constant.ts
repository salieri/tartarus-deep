import { Symbol } from './symbol';


export class Constant extends Symbol {
  public canOptimize(): boolean {
    return false;
  }
}
