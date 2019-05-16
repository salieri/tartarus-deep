import { JoiEx, JoiExSchema } from './joi-ex';

/* eslint-disable @typescript-eslint/no-empty-interface */
export interface Parameters {}


export abstract class Parameterized<TInput extends Parameters, TCoerced extends Parameters = TInput> {
  protected readonly params: TCoerced;

  public constructor(params: TInput = {} as any) {
    this.params = this.validateParams(params);
  }


  public abstract getParamSchema(): JoiExSchema;


  public validateParams(params: TInput): TCoerced {
    const result = JoiEx.validate(params, this.getParamSchema());

    if (result.error) {
      throw result.error;
    }

    return result.value as unknown as TCoerced;
  }
}

