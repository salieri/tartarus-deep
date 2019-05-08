import { Initializer, InitializerDescriptor, InitializerParams } from './initializer';
import { NDArray } from '../../math';
import { JoiEx } from '../../util';
import { Randomizer } from '../../math/randomizer';


export class RandomUniform extends Initializer {
  public constructor(params: InitializerParams = {}) {
    super(params);
  }


  public initialize(data: NDArray): NDArray {
    const randomizer: Randomizer = this.params.randomizer || this.layer.getRandomizer();

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
