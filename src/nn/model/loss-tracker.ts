export interface LossResult {
  average: number;
  final: number;
  first: number;
  min: number;
  max: number;
  samples: number;
}


export class LossTracker {
  protected samples = 0;

  protected final = Number.MAX_VALUE;

  protected min = Number.MAX_VALUE;

  protected max = Number.MAX_VALUE;

  protected first?: number = undefined;

  protected total: number = 0;

  public record(loss: number): void {
    this.samples += 1;
    this.total += loss;

    if (this.first === undefined) {
      this.first = loss;
    }

    this.min = Math.min(this.min, loss);
    this.max = Math.max(this.max, loss);
  }


  public getSampleCount(): number {
    return this.samples;
  }


  public get(): LossResult {
    if (this.samples === 0) {
      throw new Error('No recorded samples');
    }

    return {
      average: this.total / this.samples,
      final: this.final,
      first: this.first as number,
      min: this.min,
      max: this.max,
      samples: this.samples,
    };
  }
}

