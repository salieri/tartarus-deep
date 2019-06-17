import _ from 'lodash';

import {
  IndexResult,
  NDArray,
  NDArrayConstructorType,
  NDArrayPosition,
  NumberTreeElement,
} from './ndarray';

import { Matrix } from './matrix';

export enum VectorDirection {
  Row,
  Col
}


export class Vector extends NDArray {
  /* constructor(...dimensions: any[]) {
    super(...dimensions);
  } */


  protected validateConstructor(dimensions: NDArrayConstructorType): void {
    super.validateConstructor(dimensions);

    if (dimensions.length === 1) {
      const dimEl = dimensions[0];

      if (dimEl instanceof NDArray) {
        if (dimEl.countDims() !== 1) {
          throw new Error('Vector must have exactly one data dimension');
        }
      } else if (_.isArray(dimEl) === true) {
        if (Vector.resolveDimensions(dimEl as NumberTreeElement).length !== 1) {
          throw new Error('Vector must have exactly one data dimension');
        }
      }
    }
  }


  /**
   * Get vector size
   */
  public getSize(): number {
    return this.dimensions[0];
  }


  /**
   * Expand vector into a matrix of the specified size, cloning the context of the vector on each row or column
   * @param {int} rows
   * @param {int} cols
   * @param {String} direction 'row' or 'col'
   * @returns { Matrix }
   */
  public expandToMatrix(rows: number, cols: number, direction: VectorDirection): Matrix {
    const result: Matrix = new Matrix(rows, cols);

    if (
      ((direction === VectorDirection.Row) && (rows !== this.getSize()))
      || ((direction === VectorDirection.Col) && (cols !== this.getSize()))
    ) {
      throw new Error(`Vector does not fit the shape of the matrix (matrix: [${rows}, ${cols}], size: ${this.getSize()})`);
    }

    let tPos = 0;

    for (let y = 0; y < rows; y += 1) {
      if (direction === VectorDirection.Row) {
        tPos = y;
      }

      for (let x = 0; x < cols; x += 1) {
        if (direction === VectorDirection.Col) {
          tPos = x;
        }

        result.setAt([y, x], this.getAt([tPos]));
      }
    }

    return result;
  }


  /**
   * Calculate dot product between two vectors
   * @param { Vector } b
   */
  public dot(b: Vector): number {
    let total = 0;

    this.traverse(
      (aVal: number, path: number[]) => {
        const bVal: number = b.getAt(path);

        total += aVal * bVal;
      },
    );

    return total;
  }


  /**
   * Calculate Euclidean length of the vector
   */
  public length(): number {
    return Math.sqrt(this.sum((val: number): number => (val ** 2)));
  }


  /**
   * Spread values diagonally
   */
  public diagonal(): Matrix {
    const m = new Matrix(this.getSize(), this.getSize());

    this.traverse(
      (val: number, position: NDArrayPosition) => {
        m.setAt([position[0], position[0]], val);
      },
    );

    return m;
  }


  public slice(pos: NDArrayPosition, size: number): Vector {
    const nd = new Vector(size);

    _.times(size, (n: number) => nd.setAt([n], this.getAt([pos[0] + n])));

    return nd;
  }


  /**
   * Test whether `index` is in top `k`
   * @param index
   * @param k
   */
  public inTopK(index: number, k: number): boolean {
    return !!_.find(this.topK(k), (v: IndexResult) => (v.index === index));
  }


  /**
   * Get top `k` values and indexes
   * @param k
   */
  public topK(k: number): IndexResult[] {
    const values = _.map(
      this.data as number[],
      (value: number, index: number): IndexResult => ({ index, value }),
    );

    const sortedValues = _.sortBy(values, (v: IndexResult) => v.value);

    return _.reverse(_.slice(sortedValues, sortedValues.length - k));
  }


  /**
   * Get the index of the highest value in the array
   */
  public argmax(): NDArrayPosition {
    let index = null;
    let knownMax: number|null = null;

    this.traverse(
      (value, position) => {
        if ((knownMax === null) || (value > knownMax)) {
          knownMax = value;
          index = position;
        }
      },
    );

    if (index === null) {
      throw new Error('Vector has no values');
    }

    return [index];
  }


  /**
   * Get the index of the lowest value in the array
   */
  public argmin(): NDArrayPosition {
    let index = null;
    let knownMin: number|null = null;

    this.traverse(
      (value, position) => {
        if ((knownMin === null) || (value < knownMin)) {
          knownMin = value;
          index = position;
        }
      },
    );

    if (index === null) {
      throw new Error('Vector has no values');
    }

    return index;
  }


  /**
   * Outer product
   * u (x) v, where u = this
   */
  public outer(v: Vector): Matrix {
    /* tslint:disable-next-line */
    const u = this;
    const m = new Matrix(u.getSize(), v.getSize());

    for (let ui = 0; ui < u.getSize(); ui += 1) {
      for (let vj = 0; vj < v.getSize(); vj += 1) {
        m.setAt([ui, vj], u.getAt(ui) * v.getAt(vj));
      }
    }

    return m;
  }


  protected instantiate<T extends NDArray>(this: T, ...dimensions: number[]): T {
    return new Vector(...dimensions) as unknown as T;
  }
}
