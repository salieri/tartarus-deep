import {
  JoiEx,
  JoiExSchema,
  Model,
  NDArray,
  Parameterized,
  Parameters,
} from '../src';


export interface SampleData {
  x: NDArray;
  y: NDArray;
}


export interface SampleGeneratorParams extends Parameters {
  seed?: string;
}


export abstract class SampleGenerator extends Parameterized<SampleGeneratorParams> {
  public constructor(params: SampleGeneratorParams = {}) {
    super(params);
  }

  public abstract async model(): Promise<Model>;

  public abstract async samples(count: number): Promise<SampleData[]>;


  public getParamSchema(): JoiExSchema {
    return JoiEx.object().keys(
      {
        seed: JoiEx.string().optional().default('test-1234'),
      },
    );
  }
}
