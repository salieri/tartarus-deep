import _ from 'lodash';
import { NDArray } from '../../../math';
import { ValueNotSetError, ValueNotDeclaredError, InvalidValueError } from '../../../error';

export type DeferredValueType = NDArray;

export class DeferredValue {
  private value: DeferredValueType|null = null;

  private dimensions: number[]|null = null;


  public constructor(dimensions?: number[]|number) {
    if (typeof dimensions !== 'undefined') {
      this.declare(dimensions);
    }
  }


  public declare(dimensions: number[]|number): void {
    if (this.dimensions !== null) {
      throw new Error('Value dimensions have already been declared');
    }

    this.dimensions = _.castArray(dimensions);
  }


  public set(value: NDArray): void {
    this.mustBeDeclared();

    if (!_.isEqual(this.dimensions, value.getDims())) {
      throw new InvalidValueError('Value does not match expected dimensions');
    }

    this.value = value;
  }


  public countElements(): number {
    this.mustBeDeclared();

    return _.reduce(
      this.dimensions,
      (total, dimensionSize) => (total * dimensionSize),
      1,
    );
  }


  public getDims(): number[] {
    this.mustBeDeclared();

    return _.cloneDeep(this.dimensions as number[]);
  }


  public get(): DeferredValueType {
    this.mustBeDeclared();

    if (!this.value) {
      throw new ValueNotSetError('Value has not been set');
    }

    return this.value;
  }


  private mustBeDeclared(): void {
    if (this.dimensions === null) {
      throw new ValueNotDeclaredError('Value has not been declared yet');
    }
  }
}

