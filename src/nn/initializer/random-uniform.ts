import { Initializer, InitializerParams } from './initializer';
import { NDArray } from '../../math';
import { JoiEx, JoiExSchema } from '../../util';
import { Randomizer } from '../../math/randomizer';


export interface RandomUniformParams extends InitializerParams {
  min?: number;
  max?: number;
  randomizer?: Randomizer|null;
}

export class RandomUniform extends Initializer<RandomUniformParams> {
  protected readonly min: number;

  protected readonly max: number;


  public constructor(params: RandomUniformParams = {}) {
    super(params);

    this.min = this.params.min || 0.0;
    this.max = this.params.max || 1.0;
  }


  private getRandomizer(): Randomizer {
    if (this.params.randomizer) {
      return this.params.randomizer as Randomizer;
    }

    if (!this.layer) {
      throw new Error('This initializer is not attached to a layer; randomizer must be passed in constructor');
    }

    return this.layer.getSession().getRandomizer();
  }


  public async initialize(data: NDArray): Promise<NDArray> {
    const randomizer: Randomizer = this.getRandomizer();

    return data.apply((): number => (randomizer.floatBetween(this.min, this.max)));
  }


  public getParamSchema(): JoiExSchema {
    return JoiEx.object().keys(
      {
        min: JoiEx.number().default(0.0).description('Minimum random value'),
        max: JoiEx.number().default(1.0).description('Maximum random value'),
        randomizer: JoiEx.random().description('Randomizer').default(null),
      },
    );
  }
}
