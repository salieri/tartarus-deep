import { NDArray } from '../../math';
import { JoiEx } from '../../util';
import { NDSymbol, SymbolCollection } from '../symbol';


export interface LayerParams {
  [key: string]: any;
}

export interface LayerDescriptor {
  [key: string]: any;
}


export abstract class Layer {
  public params: LayerParams;

  public name: string;

  protected compiled: boolean = false;

  private static layerCounter: number = 0;


  public constructor(params: LayerParams = {}, name?: string) {
    this.params = this.validateParams(params);

    Layer.layerCounter += 1;

    this.name = name || `${this.constructor.name}#${Layer.layerCounter}`;
  }


  private validateParams(params: LayerParams): LayerParams {
    const result = JoiEx.validate(params, this.getDescriptor());

    if (result.error) {
      throw result.error;
    }

    return result.value;
  }


  public setParam(paramName: string, value: any): Layer {
    const result = JoiEx.validate(this.params[paramName], value);

    if (result.error) {
      throw result.error;
    }

    this.params[paramName] = result.value;

    return this;
  }


  public getDescriptor(): LayerDescriptor {
    return {};
  }


  public calculate(x: NDArray): NDArray {
    return x;
  }


  public forward(input: NDArray) {
  }


  public backward(output: NDArray) {
  }


  public getSymbol(name: string): any {
  }


  public setSymbol(name: string, symbol: Symbol) {

  }

  public hasSymbol(name: string) {

  }


  protected mustHaveSymbol(name: string): void {

  }


  public register(variableName: string, symbol: NDSymbol): void {
    this.symbols.add(this.getSymbolName(variableName), symbol);
  }


  public getSymbolName(variableName: string): string {
    return `${this.getLayerName()}-${variableName}`;
  }


  public getLayerName(): string {
    return this.name;
  }


  protected canModify(): void {
    if (this.compiled) {
      throw new Error('Layer cannot be modified after compilation');
    }
  }


  public compile(): void {
    this.canModify();

    this.compiled = true;
  }
}

