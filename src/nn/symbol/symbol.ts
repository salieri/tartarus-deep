import { NDArray } from '../../math';


export class Symbol {
  protected name: string;
  protected data: NDArray;


  constructor(name: string, data: NDArray) {
    this.name = name;
    this.data = data;
  }


  public canOptimize(): boolean {
    throw new Error('Not implemented');
  }


  public get(): NDArray {
    return this.data;
  }


  public set(data: NDArray) {
    this.data = data;
  }
}
