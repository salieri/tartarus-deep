import _ from 'lodash';
import sha256 from 'fast-sha256';

/**
 * Pseudo random number generator
 */
export abstract class Randomizer {
  protected seed: string;

  protected seedHash: Uint8Array;

  protected seedPtr: number = 0;

  public constructor(seed?: string) {
    const enc = new TextEncoder();

    this.seed = _.toString(seed || Randomizer.generateSeed());
    this.seedHash = sha256(enc.encode(this.seed));
  }


  private static generateSeed(): string {
    const d = new Date();

    return `Moo#${d.getTime()}#${Math.random()}`;
  }


  protected readIntFromSeed(): number {
    const ptrBase: number = this.seedPtr * 4;

    const val: number = (this.seedHash[ptrBase] << 24)
      + (this.seedHash[ptrBase] << 16)
      + (this.seedHash[ptrBase] << 8)
      + (this.seedHash[ptrBase]);

    this.seedPtr += 1;

    return val;
  }


  public getSeed(): string {
    return this.seed;
  }


  /**
   * Return a random number between 0 (inclusive) and 1 (exclusive)
   */
  public abstract random(): number;


  /**
   * Return a random `int` between 0 (inclusive) and `range` (exclusive)
   */
  public range(range: number): number {
    return Math.round(this.random() * range);
  }


  /**
   * Return a random `int` between `inclusiveMin` (inclusive) and `exclusiveMax` (exclusive)
   */
  public intBetween(inclusiveMin: number, exclusiveMax: number): number {
    return Math.round(this.floatBetween(inclusiveMin, exclusiveMax));
  }


  /**
   * Return a random number between `inclusiveMin` (inclusive) and `exclusiveMax` (exclusive)
   */
  public floatBetween(inclusiveMin: number, exclusiveMax: number): number {
    return inclusiveMin + this.range(exclusiveMax - inclusiveMin);
  }
}

