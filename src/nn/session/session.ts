import { Xoshiro128, Randomizer } from '../../math/randomizer';
import { MuteLogger, Logger } from '../../util';


export class Session {
  private randomizer: Randomizer;

  private logger: Logger;

  public constructor(seed?: string) {
    this.randomizer = new Xoshiro128(seed);
    this.logger = new MuteLogger(); // ConsoleLogger();
  }


  public random(): number {
    return this.randomizer.random();
  }


  public getRandomizer(): Randomizer {
    return this.randomizer;
  }


  public getLogger(): Logger {
    return this.logger;
  }
}

