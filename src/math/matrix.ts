import _ from 'lodash';
import { NDArray, NDArrayConstructorType } from './ndarray';
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
        if (Matrix.resolveDimensions(dimEl).length !== 2) {
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
   * Clone a matrix
   * @param { Matrix } [targetObj=null]
   * @returns { Matrix }
   * @public
   */
  public clone(targetObj: Matrix): Matrix {
    const to = targetObj || new Matrix(...this.dimensions);

    return super.clone(to) as Matrix;
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
}

