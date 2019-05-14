import { Initializer, InitializerDescriptor, InitializerParams } from './initializer';
import { NDArray } from '../../math';
import { JoiEx } from '../../util';
import { Randomizer } from '../../math/randomizer';


export class RandomUniform extends Initializer {
  public constructor(params: InitializerParams = {}) {
    super(params);
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

    return data.apply((): number => (randomizer.floatBetween(this.params.min, this.params.max)));
  }


  public getDescriptor(): InitializerDescriptor {
    return {
      min: JoiEx.number().default(0.0).description('Minimum random value'),
      max: JoiEx.number().default(1.0).description('Maximum random value'),
      randomizer: JoiEx.random().description('Randomizer').default(null),
    };
  }
}
