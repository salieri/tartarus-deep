import {
  DeferredInputFeed,
  JoiEx,
  JoiExSchema,
  Model,
  Parameterized,
  Parameters,
} from '../src';


export interface SampleGeneratorParams extends Parameters {
  seed?: string;
}


export abstract class SampleGenerator extends Parameterized<SampleGeneratorParams> {
  public constructor(params: SampleGeneratorParams = {}) {
    super(params);
  }

  public abstract model(): Model;

  public abstract samples(count: number): DeferredInputFeed;


  public getParamSchema(): JoiExSchema {
    return JoiEx.object().keys(
      {
        seed: JoiEx.string().optional().default('test-1234'),
      },
    );
  }
}
