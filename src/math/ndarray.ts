import _ from 'lodash';
import { Randomizer } from './randomizer';

export type NumberTreeElement = number[] | number;
export type NDArrayConstructorType = NumberTreeElement[]|NDArray[]|number[]|[number[]]|[number[][]]|[number[][][]];

export type NDArrayPosition = number[];

export type NDArrayTraverseElementCallback = (value: number, position: NDArrayPosition) => number|undefined|void;
export type NDArrayTraverseArrayCallback = (branch: NumberTreeElement[], position: NDArrayPosition) => void;
export type NDArrayElementwiseOpCallback = (a: number, b:number) => number;
export type NDArraySummingCallback = (value: number) => number;
export type NDArrayIterationCallback = (arrayValues: number[], position: NDArrayPosition) => number;


export interface IndexResult {
  value: number;
  index: number;
}


export interface NDArrayCollection {
  [key: string]: NDArray;
}


export class NDArray {
  private static idCounter: number = 0;

  protected id: number = 0;

  protected dimensions: number[] = [];

  protected data: NumberTreeElement[] = [];

  /**
   * ```ts
   * // create n-dimensional array with the size of 20x30x40
   * const nd1 = new NDArray( 20, 30, 40 );
   *
   * // create a clone of `anotherNdArray`
   * const nd2 = new NDArray( anotherNdArray );
   *
   * // create one-dimensional array with the given data
   * const nd3 = new NDArray([1, 2, 3]);
   *
   * // create a n-dimensional array with the given data
   * const nd4 = new NDArray(
   *   [
   *     [ 0, 0, 0 ],
   *     [ 1, 1, 1 ],
   *     [ 2, 3, 4 ]
   *   ]
   * );
   * ```
   */
  public constructor(...dimensionsOrValue: NDArrayConstructorType) {
    this.validateConstructor(dimensionsOrValue);

    if (dimensionsOrValue.length === 1) {
      const dimEl = dimensionsOrValue[0];

      if (dimEl instanceof NDArray) {
        dimEl.clone(this);
        return;
      }

      if (_.isArray(dimEl) === true) {
        this.dimensions = NDArray.resolveDimensions(dimEl as NumberTreeElement);
        this.data = NDArray.createDimArray(this.dimensions);

        this.setData(dimEl as NumberTreeElement[]);
        return;
      }
    }

    this.dimensions = dimensionsOrValue as number[];
    this.data = NDArray.createDimArray(this.dimensions);

    NDArray.idCounter += 1;

    this.id = NDArray.idCounter;
  }


  protected validateConstructor(dimensions: NDArrayConstructorType): void {
    if ((!dimensions) || (dimensions.length === 0)) {
      throw new Error('Unspecified dimensions');
    }
  }


  /**
   * @param dimensions
   * @param initialValue
   * @returns Multi-dimensional array representing the data
   */
  public static createDimArray(dimensions: number[], initialValue: number = 0): NumberTreeElement[] {
    function iterateCreation(depthIndex: number): NumberTreeElement[] {
      const size = dimensions[depthIndex];

      if (depthIndex === dimensions.length - 1) {
        return _.fill(Array(size), initialValue);
      }

      return _.map(Array(size), () => iterateCreation(depthIndex + 1)) as NumberTreeElement[];
    }

    return iterateCreation(0);
  }


  protected static resolveDimensions(data: NumberTreeElement): number[] {
    const dimensions = [];

    let dp = data;

    while (_.isArray(dp) === true) {
      dp = dp as number[];

      dimensions.push(dp.length);

      dp = dp[0];
    }

    return dimensions;
  }


  /**
   * Traverse all elements and arrays in the NDArray
   * @param depthIndex
   * @param positionPath
   * @param dataSource
   * @param dimensions
   * @param elementCallback
   * @param arrayCallback
   */
  private static traverseNDArray(
    depthIndex        : number,
    positionPath      : NDArrayPosition,
    dataSource        : NumberTreeElement[],
    dimensions        : number[],
    elementCallback?  : NDArrayTraverseElementCallback,
    arrayCallback?    : NDArrayTraverseArrayCallback,
  ): void {
    positionPath.push(0);

    const posIdx  = positionPath.length - 1;
    const dsArray = dataSource as number[];

    _.each(
      dsArray,
      (dataVal: NumberTreeElement, dataIdx: number) => {
        /* eslint-disable-next-line */
        positionPath[posIdx] = dataIdx;

        if (depthIndex === dimensions.length - 1) {
          if (elementCallback) {
            const result = elementCallback(dataVal as number, positionPath);

            if (_.isUndefined(result) === false) {
              dsArray[dataIdx] = result as number;
            }
          }
        } else {
          if (arrayCallback) {
            arrayCallback(dataVal as NumberTreeElement[], positionPath);
          }

          NDArray.traverseNDArray(depthIndex + 1, positionPath, dataVal as number[], dimensions, elementCallback, arrayCallback);
        }
      },
    );

    positionPath.pop();
  }


  /**
   * Traverse all arrays in the multi-dimensional array
   * @param callback
   */
  public traverseArrays(callback: NDArrayTraverseArrayCallback): void {
    NDArray.traverseNDArray(0, [], this.data, this.dimensions, undefined, callback);
  }


  /**
   * Traverse all elements in the multi-dimensional array
   *
   * @param callback Called for each element in the multi-dimensional array.
   *  If the function returns a value other than undefined, the element will be set to that value.
   */
  public traverse(callback: NDArrayTraverseElementCallback): void {
    NDArray.traverseNDArray(0, [], this.data, this.dimensions, callback, undefined);
  }


  /**
   * Clone an NDArray
   */
  public clone(targetObj?: NDArray): NDArray {
    /* eslint-disable-next-line */
    const target = (targetObj || new NDArray(...this.dimensions)) as any;

    _.each(
      this,
      (v, k) => {
        target[k] = _.cloneDeep(v);
      },
    );

    NDArray.idCounter += 1;

    target.id = NDArray.idCounter;

    return target;
  }


  /**
   * Randomize all values
   * @param min (inclusive)
   * @param max (exclusive)
   * @param randomizer Randomizer instance to use; uses Math.random() if none provided
   */
  public rand(min: number = 0.0, max: number = 1.0, randomizer?: Randomizer): NDArray {
    return this.apply(
      (): number => (randomizer ? randomizer.floatBetween(min, max) : (min + Math.random() * (max - min))),
    );
  }


  /**
   * Set all values to zero
   */
  public zero(): NDArray {
    return this.set(0);
  }


  /**
   * Set all values to the specified value
   */
  public set(value: number): NDArray {
    return this.apply(() => (value));
  }


  /**
   * Get NDArray data
   */
  public get(): NumberTreeElement[] {
    return _.cloneDeep(this.data as number[]);
  }


  /**
   * Get value from specific position
   */
  public getAt(positionPath: NDArrayPosition|number): number {
    const finalPosition = _.castArray(positionPath);

    this.validatePosition(finalPosition);

    return _.get(this.data, _.join(finalPosition, '.'));
  }


  /**
   * Set value at specific position
   */
  public setAt(positionPath: NDArrayPosition|number, value: number): void {
    const finalPosition = _.castArray(positionPath);

    this.validatePosition(finalPosition);

    _.set(this.data, _.join(finalPosition, '.'), value);
  }


  /**
   * Set NDArray data
   */
  public setData(data: NumberTreeElement[]): void {
    if (data.length !== this.dimensions[0]) {
      throw new Error('Inconsistent data size');
    }

    NDArray.traverseNDArray(
      0,
      [],
      data,
      this.dimensions,
      undefined,
      (a: NumberTreeElement[], positionPath: NDArrayPosition) => {
        if (a.length !== this.dimensions[positionPath.length]) {
          throw new Error('Inconsistent data size');
        }
      },
    );

    this.traverse(
      (v: number, positionPath: NDArrayPosition) => {
        const newVal = _.get(data, _.join(positionPath, '.'), '### not found ###');

        if (newVal === '### not found ###') {
          throw new Error('Invalid data shape');
        }

        return newVal;
      },
    );
  }


  /**
   * Validate position path
   */
  protected validatePosition(positionPath: NDArrayPosition): void {
    // if (positionPath.length === 0) {}

    if (positionPath.length !== this.dimensions.length) {
      throw new Error(`Invalid position path: expected ${this.dimensions.length} dimensions`);
    }

    _.each(
      positionPath,
      (posIdx, dimIdx) => {
        if ((posIdx < 0) || (posIdx >= this.dimensions[dimIdx])) {
          throw new Error(
            `Invalid position path: Dimension ${dimIdx} position should be 0-${this.dimensions[dimIdx] - 1}, was ${posIdx}`,
          );
        }
      },
    );
  }


  /**
   * Returns number of dimensions in NDArray
   */
  public countDims(): number {
    return this.dimensions.length;
  }


  /**
   * Returns dimension sizes in NDArray
   */
  public getDims(): number[] {
    return _.cloneDeep(this.dimensions);
  }


  /**
   * Count all elements in NDArray
   */
  public countElements(): number {
    return _.reduce(
      this.dimensions,
      (dims, accumulator) => (dims * accumulator),
      1,
    );
  }


  /**
   * Compare with another NDArray
   */
  public equals(b: NDArray): boolean {
    if (b.countDims() !== this.countDims()) {
      return false;
    }

    return _.isEqual(this.data, b.data);
  }


  public toString(): string {
    return `NDArray#${this.id}: ${this.data.toString()}`;
  }


  public toJSON(): NumberTreeElement[] {
    return _.cloneDeep(this.data);
  }


  public apply(valCallback: NDArrayTraverseElementCallback): NDArray {
    const clone = this.clone();

    clone.traverse(valCallback);

    return clone;
  }


  /* --------------------------- Math shortcuts --------------------------- */

  /**
   * Elementwise operation between NDArrays of the same size
   * @param b
   * @param operationCb
   * @param opName
   * @protected
   */
  protected elementwiseOp(b: NDArray | number, operationCb: NDArrayElementwiseOpCallback, opName: string): NDArray {
    if (b instanceof NDArray) {
      if (!_.isEqual(this.getDims(), b.getDims())) {
        throw new Error(`Cannot do elementwise ${opName} on NDArrays with differing dimensions (a: ${this.getDims()}, b: ${b.getDims()})`);
      }
    }

    const aClone = this.clone();

    aClone.traverse(
      (val: number, pos: NDArrayPosition): number => {
        let bVal: number;

        if (b instanceof NDArray) {
          bVal = b.getAt(pos);
        } else {
          bVal = b;
        }

        return operationCb(val, bVal);
      },
    );

    return aClone;
  }


  /**
   * Elementwise subtract
   * @param subtrahend Value(s) to be subtracted
   */
  public sub(subtrahend: NDArray | number): NDArray {
    return this.elementwiseOp(
      subtrahend,
      (aVal: number, bVal: number): number => (aVal - bVal),
      'subtraction',
    );
  }


  /**
   * Elementwise addition
   * @param addend Value(s) to be added
   */
  public add(addend: NDArray | number): NDArray {
    return this.elementwiseOp(
      addend,
      (aVal: number, bVal: number): number => (aVal + bVal),
      'addition',
    );
  }


  /**
   * Elementwise multiplication
   * @param multiplicand Value(s) to be multiplied by
   */
  public mul(multiplicand: NDArray | number): NDArray {
    return this.elementwiseOp(
      multiplicand,
      (aVal: number, bVal: number): number => (aVal * bVal),
      'multiplication',
    );
  }


  /**
   * Elementwise division
   * @param divisor Value(s) to be divided by
   */
  public div(divisor: NDArray | number): NDArray {
    return this.elementwiseOp(
      divisor,
      (aVal: number, bVal: number): number => (aVal / bVal),
      'division',
    );
  }


  /**
   * Elementwise exponent (element^exponent)
   * @param exponent Value(s) to be exponented by
   */
  public pow(exponent: NDArray | number): NDArray {
    return this.elementwiseOp(
      exponent,
      (aVal: number, bVal: number): number => (aVal ** bVal),
      'exponentation',
    );
  }


  /**
   * Elementwise abs
   */
  public abs(): NDArray {
    return this.apply(Math.abs);
  }


  /**
   * Elementwise log
   */
  public log(): NDArray {
    return this.apply(Math.log);
  }


  /**
   * Elementwise exp
   */
  public exp(): NDArray {
    return this.apply(Math.exp);
  }


  /**
   * Elementwise negate values
   */
  public neg(): NDArray {
    return this.apply((val: number): number => (-val));
  }


  /**
   * Elementwise sin
   */
  public sin(): NDArray {
    return this.apply(Math.sin);
  }


  /**
   * Elementwise cos
   */
  public cos(): NDArray {
    return this.apply(Math.cos);
  }


  /**
   * Elementwise tan
   */
  public tan(): NDArray {
    return this.apply(Math.tan);
  }


  /**
   * Elementwise tanh
   */
  public tanh(): NDArray {
    return this.apply(Math.tanh);
  }


  /**
   * Elementwise arc sin
   */
  public asin(): NDArray {
    return this.apply(Math.asin);
  }


  /**
   * Elementwise arc cos
   */
  public acos(): NDArray {
    return this.apply(Math.acos);
  }


  /**
   * Elementwise arc tan
   */
  public atan(): NDArray {
    return this.apply(Math.atan);
  }


  /**
   * Elementwise square root
   */
  public sqrt(): NDArray {
    return this.apply(Math.sqrt);
  }


  /**
   * Elementwise round
   */
  public round(): NDArray {
    return this.apply(Math.round);
  }


  /**
   * Elementwise binary equals
   * Resulting NDArray contains 1s where both arrays equal, 0s otherwise
   */
  public equal(anotherArray: NDArray): NDArray {
    return NDArray.iterate(
      (values: number[]): number => (values[0] === values[1] ? 1 : 0),
      this,
      anotherArray,
    );
  }


  /**
   * Get the minimum value in the array
   */
  public min(): number {
    let knownMin: number|null = null;

    this.traverse(
      (value) => {
        if ((knownMin === null) || (value < knownMin)) {
          knownMin = value;
        }
      },
    );

    if (knownMin === null) {
      throw new Error('NDArray has no values');
    }

    return knownMin;
  }


  /**
   * Get the maximum value in the array
   */
  public max(): number {
    let knownMax: number|null = null;

    this.traverse(
      (value) => {
        if ((knownMax === null) || (value > knownMax)) {
          knownMax = value;
        }
      },
    );

    if (knownMax === null) {
      throw new Error('NDArray has no values');
    }

    return knownMax;
  }


  /**
   * Elementwise clamping
   */
  public clamp(minVal: number, maxVal: number): NDArray {
    return this.apply(
      (v: number): number => Math.min(Math.max(v, minVal), maxVal),
    );
  }


  /**
   * Calculate total sum of elements
   * @param summingCallback If defined, the return value of this function is used instead of the element value
   */
  public sum(summingCallback?: NDArraySummingCallback): number {
    let total = 0;

    this.traverse(
      (val: number): void => {
        if (summingCallback) {
          total += summingCallback(val);
        } else {
          total += val;
        }
      },
    );

    return total;
  }


  /**
   * Normalize elements
   */
  public normalize(): NDArray {
    const expSum = Math.sqrt(this.sum((val: number): number => (val ** 2)));

    return this.apply((val: number): number => (val / expSum));
  }


  /**
   * Calculate mean of NDArray
   */
  public mean(): number {
    return this.sum() / this.countElements();
  }


  /**
   * Flatten into one-dimensional NDArray
   */
  public flatten(): NDArray {
    const nd = new NDArray(this.countElements());

    nd.setData(_.flatten(this.data));

    return nd;
  }


  public getId(): number {
    return this.id;
  }


  protected static copyDataEx(target: NDArray, source: NDArray, targetPosStart: number, sourcePosStart: number, count: number): void {
    const sourcePos = [sourcePosStart];
    const targetPos = [targetPosStart];

    for (let pos = 0; pos < count; pos += 1) {
      target.setAt(targetPos, source.getAt(sourcePos));

      sourcePos[0] += 1;
      targetPos[0] += 1;
    }
  }


  /**
   * Concatenate NDArray with another NDArray
   * The resulting NDArray is flattened into a one-dimensional array.
   */
  public concat(anotherArray: NDArray): NDArray {
    const myNd = this.flatten();
    const anotherNd = anotherArray.flatten();

    const myElementCount = myNd.countElements();
    const anotherElementCount = anotherNd.countElements();

    const nd = new NDArray(myElementCount + anotherElementCount);

    NDArray.copyDataEx(nd, myNd, 0, 0, myElementCount);
    NDArray.copyDataEx(nd, anotherNd, myElementCount, 0, anotherElementCount);

    return nd;
  }


  /**
   * Iterate multiple NDArrays of the same shape and provide the element value from each NDArray to the callback.
   * Returns a new NDArray of the same shape with its values set to the return values of the callback
   * @param callback
   * @param arrays
   */
  public static iterate(callback: NDArrayIterationCallback, ...arrays: NDArray[]): NDArray {
    const clone = arrays[0].clone();

    clone.traverse(
      (val: number, pos: NDArrayPosition): number => {
        const vals: number[] = _.map(
          arrays,
          (arr: NDArray) => (arr.getAt(pos)),
        );

        return callback(vals, pos);
      },
    );

    return clone;
  }
}

