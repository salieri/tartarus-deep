import _ from 'lodash';
import { Matrix, NDArray, Vector } from '../../../math';
import { ValueNotSetError, ValueNotDeclaredError, InvalidValueError } from '../../../error';


export class DeferredValue {
  private static idCounter: number = 0;

  private value: NDArray|null = null;

  private dimensions: number[]|null = null;

  private id: number;


  public constructor(dimensions?: number[]|number) {
    if (typeof dimensions !== 'undefined') {
      this.declare(dimensions);
    }

    DeferredValue.idCounter += 1;

    this.id = DeferredValue.idCounter;
  }


  public declare(dimensions: number[]|number): void {
    if (this.dimensions !== null) {
      throw new Error('Value dimensions have already been declared');
    }

    this.dimensions = _.castArray(dimensions);
  }


  protected castType<NDType extends NDArray = NDArray>(value: NDArray|Matrix|Vector, type?: { new(v: any): NDType }): NDType {
    const finalType = type || NDArray;

    if (value instanceof finalType) {
      return value as NDType;
    }

    /* eslint-disable-next-line new-cap */
    return new finalType(value) as NDType;
  }


  public set<NDType>(value: NDArray|Matrix|Vector, type?: { new (val: NDArray|Matrix|Vector): NDType }): void {
    this.mustBeDeclared();

    if (!_.isEqual(this.dimensions, value.getDims())) {
      throw new InvalidValueError(`Value does not match expected dimensions (expected: ${this.dimensions}, given: ${value.getDims()})`);
    }

    this.value = this.castType(value, type as any);
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


  public get<NDType extends NDArray = NDArray>(type?: { new (val: NDArray|Matrix|Vector): NDType }): NDType {
    this.mustBeDeclared();

    if (!this.value) {
      throw new ValueNotSetError('Value has not been set');
    }

    return this.castType(this.value, type as any) as unknown as NDType;
  }


  private mustBeDeclared(): void {
    if (this.dimensions === null) {
      throw new ValueNotDeclaredError('Value has not been declared yet');
    }
  }


  public isSet(): boolean {
    return !!this.value;
  }


  public unset(): void {
    this.value = null;
  }
}

