import _ from 'lodash';
import { NDArray, NDArrayConstructorType, NumberTreeElement } from './ndarray';
import { Vector } from './vector';


export class Matrix extends NDArray {
  /* public constructor(...dimensions: any[]) {
    super(...dimensions);
  } */


  protected validateConstructor(dimensions: NDArrayConstructorType): void {
    super.validateConstructor(dimensions);

    if (dimensions.length === 1) {
      const dimEl = dimensions[0];

      if (dimEl instanceof NDArray) {
        if (dimEl.countDims() !== 2) {
          throw new Error('Matrix must have exactly two data dimensions');
        }
      } else if (_.isArray(dimEl) === true) {
        if (Matrix.resolveDimensions(dimEl as NumberTreeElement).length !== 2) {
          throw new Error('Matrix must have exactly two data dimensions');
        }
      }
    }
  }


  /**
   * @public
   * @returns {int}
   */
  public getCols(): number {
    return this.dimensions[1];
  }


  /**
   * @public
   * @returns {int}
   */
  public getRows(): number {
    return this.dimensions[0];
  }


  /**
   * @public
   * @return { Matrix }
   */
  public transpose(): Matrix {
    const result = new Matrix(this.getCols(), this.getRows());

    for (let y = 0; y < this.getRows(); y += 1) {
      for (let x = 0; x < this.getCols(); x += 1) {
        result.setAt([x, y], this.getAt([y, x]));
      }
    }

    return result;
  }


  /**
   * Matrix multiplication
   */
  public matmul(b: Matrix): Matrix {
    const aCols: number = this.getCols();
    const aRows: number = this.getRows();
    const bCols: number = b.getCols();
    const bRows: number = b.getRows();

    if (aCols !== bRows) {
      throw new Error('Cannot multiply matrices where a.cols does not match b.rows');
    }

    const result: Matrix = new Matrix(aRows, bCols);

    for (let y = 0; y < aRows; y += 1) {
      for (let x = 0; x < bCols; x += 1) {
        let val = 0;

        for (let i = 0; i < aCols; i += 1) {
          val += this.getAt([y, i]) * b.getAt([i, x]);
        }

        result.setAt([y, x], val);
      }
    }

    return result;
  }


  public dot(b: Matrix|Vector): Matrix|Vector {
    const dims = b.getDims();

    if (dims.length === 1) {
      return this.vecmul(new Vector(b));
    }

    return this.matmul(new Matrix(b));
  }


  /**
   * Multiply matrix by a vector
   */
  public vecmul(b: Vector): Vector {
    // const aCols: number = this.getCols();
    const aRows: number = this.getRows();
    const bSize: number = b.getSize();

    const result: Vector = new Vector(aRows);

    for (let y = 0; y < aRows; y += 1) {
      let val = 0;

      for (let x = 0; x < bSize; x += 1) {
        val += this.getAt([y, x]) * b.getAt([x]);
      }

      result.setAt([y], val);
    }

    return result;
  }


  public pickDiagonal(): Vector {
    if (this.getCols() !== this.getRows()) {
      throw new Error(
        `Cannot pick diagonally across a matrix which columns and rows are not equal size (${this.getCols()} cols, ${this.getRows()} rows)`,
      );
    }

    const v = new Vector(this.getCols());

    for (let n = 0; n < this.getCols(); n += 1) {
      v.setAt(n, this.getAt([n, n]));
    }

    return v;
  }


  protected instantiate<T extends NDArray>(this: T, ...dimensions: number[]): T {
    return new Matrix(...dimensions) as unknown as T;
  }


  public toString(): string {
    return `Matrix#${this.id}: ${this.data.toString()}`;
  }
}

