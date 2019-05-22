import { NDSymbol } from './nd-symbol';


export class Constant extends NDSymbol {
  public canOptimize(): boolean {
    return false;
  }
}
