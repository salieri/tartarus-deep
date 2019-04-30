import { NDArray } from '../../math';


export abstract class Symbol {
  protected name: string;
  protected data: NDArray;


  constructor(name: string, data: NDArray) {
    this.name = name;
    this.data = data;
  }


  public abstract canOptimize(): boolean;


  public get(): NDArray {
    return this.data;
  }


  public set(data: NDArray) {
    this.data = data;
  }
}
