import { Xoshiro128, Randomizer } from '../../math/randomizer';


export class Session {
  public randomizer: Randomizer;

  constructor(seed?: string) {
    this.randomizer = new Xoshiro128(seed);
  }


  public random(): number {
    return this.randomizer.random();
  }
}

