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

  public abstract model(): Model;

  public abstract samples(count: number): SampleData[];


  public getParamSchema(): JoiExSchema {
    return JoiEx.object().keys(
      {
        seed: JoiEx.string().optional().default('test-1234'),
      },
    );
  }
}
