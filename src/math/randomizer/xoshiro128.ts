import { Randomizer } from './randomizer';

/**
 * xoshiro128 pseudo random number generator
 * @url https://stackoverflow.com/a/47593316/844771
 */
export class Xoshiro128 extends Randomizer {
  protected a: number;

  protected b: number;

  protected c: number;

  protected d: number;

  public constructor(seed?: string) {
    super(seed);

    this.a = this.readIntFromSeed();
    this.b = this.readIntFromSeed();
    this.c = this.readIntFromSeed();
    this.d = this.readIntFromSeed();
  }


  /**
   * Return a random number between 0 (inclusive) and 1 (exclusive)
   */
  public random(): number {
    const t: number = this.b << 9;
    let r: number = this.a * 5;

    r = ((r << 7) | (r >>> 25)) * 9;

    this.c ^= this.a;
    this.d ^= this.b;

    this.b ^= this.c;
    this.a ^= this.d;
    this.c ^= t;

    this.d = (this.d << 11) | (this.d >>> 21);

    return (r >>> 0) / 4294967296;
  }
}

