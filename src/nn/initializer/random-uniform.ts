import { Initializer, InitializerDescriptor, InitializerParams } from './initializer';
import { NDArray } from '../../math';
import { JoiEx } from '../../util';
import { Randomizer, Xoshiro128 } from '../../math/randomizer';


export class RandomUniform extends Initializer {
  private randomizer: Randomizer;

  public constructor(params: InitializerParams) {
    super(params);

    this.randomizer = params.randomizer;
  }


  public initialize(data: NDArray): NDArray {
    return data.apply((): number => (this.randomizer.floatBetween(this.params.min, this.params.max)));
  }


  public getDescriptor(): InitializerDescriptor {
    return {
      min: JoiEx.number().default(0.0).description('Minimum random value'),
      max: JoiEx.number().default(1.0).description('Maximum random value'),
      randomizer: JoiEx.random().description('Randomizer').default(new Xoshiro128()),
    };
  }
}
