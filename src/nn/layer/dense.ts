import { Layer, LayerParams } from './layer';
import { Activation } from '../activation';
import { JoiEx, JoiExSchema } from '../../util';
import { Matrix, NDArray, Vector } from '../../math';
import { Initializer } from '../initializer';
import { DeferredReadonlyCollection } from '../symbols';
import { KeyNotFoundError } from '../../error';


export interface DenseParamsInput extends LayerParams {
  units: number;
  activation?: Activation|string;
  bias?: boolean;
  biasInitializer?: Initializer|string;
  weightInitializer?: Initializer|string;
}

export interface DenseParamsCoerced extends DenseParamsInput {
  activation: Activation;
  biasInitializer: Initializer;
  weightInitializer: Initializer;
}


export class Dense extends Layer<DenseParamsInput, DenseParamsCoerced> {
  protected static readonly WEIGHT_MATRIX = 'weight';

  protected static readonly BIAS_MATRIX = 'bias';

  protected static readonly LINEAR_OUTPUT = 'linear';

  protected static readonly ACTIVATED_OUTPUT = 'activated';

  protected readonly input: DeferredReadonlyCollection = new DeferredReadonlyCollection();


  /**
   * dW[L] =
   */
  protected async backwardExec(dPrev): Promise<void> {

  }


  protected async forwardExec(): Promise<void> {
    const linearOutput    = this.calculate(this.input.getDefault().get());
    const activatedOutput = this.activate(linearOutput);

    this.output.setValue(Dense.LINEAR_OUTPUT, linearOutput);
    this.output.setValue(Dense.ACTIVATED_OUTPUT, activatedOutput);
  }


  /**
   * Z = A_prev * W + b
   */
  protected calculate(input: NDArray): NDArray {
    const weight = new Matrix(this.optimizer.getValue(Dense.WEIGHT_MATRIX));

    let output = weight.vecmul(new Vector(input.flatten()));

    if (this.params.bias) {
      output = output.add(new Vector(this.optimizer.getValue(Dense.BIAS_MATRIX))) as Vector;
    }

    return output;
  }


  /**
   * A = g(Z)
   */
  protected activate(linearOutput: NDArray): NDArray {
    const activation = this.params.activation as Activation;

    return activation.calculate(linearOutput);
  }


  protected resolveInput(): void {
    try {
      const defaultInput = this.rawInputs.getDefault();

      this.input.setCollection(defaultInput);
      return;
    } catch (err) {
      if (!(err instanceof KeyNotFoundError)) {
        throw err;
      }
    }

    if (this.rawInputs.count() < 1) {
      throw new Error(`Missing input for dense layer '${this.getName()}'`);
    }

    if (this.rawInputs.count() > 1) {
      throw new Error(`Too many inputs for a dense layer '${this.getName()}'`);
    }

    this.input.setCollection(this.rawInputs.first());
  }


  protected async compileExec(): Promise<void> {
    this.resolveInput();

    this.input.requireDefault();

    const inputUnits = this.input.getDefault().countElements();
    const units = this.params.units;

    this.optimizer.declare(Dense.WEIGHT_MATRIX, [units, inputUnits]);

    if (this.params.bias) {
      this.optimizer.declare(Dense.BIAS_MATRIX, [units]);
    }

    this.output.declare(Dense.LINEAR_OUTPUT, [units]);
    this.output.declare(Dense.ACTIVATED_OUTPUT, [units]);

    this.output.setDefaultKey(Dense.ACTIVATED_OUTPUT);
  }


  protected async initializeExec(): Promise<void> {
    const wInit = this.params.weightInitializer;
    const weight = this.optimizer.get(Dense.WEIGHT_MATRIX);

    weight.set(await wInit.initialize(new NDArray(...weight.getDims())));

    if (this.params.bias) {
      const bInit = this.params.biasInitializer;
      const bias = this.optimizer.get(Dense.BIAS_MATRIX);

      bias.set(await bInit.initialize(new NDArray(...bias.getDims())));
    }
  }


  public getParamSchema(): JoiExSchema {
    return JoiEx.object().keys(
      {
        units: JoiEx.number().default(16).min(1).description('Number of outputs'),
        activation: JoiEx.activation().default('identity').description('Activation function'),

        bias: JoiEx.boolean().default(true).description('Apply bias'),
        biasInitializer: JoiEx.initializer().default('zero').description('Bias initializer'),

        // biasRegularizer : JoiEx.regularizer().default( null ).description( 'Bias regularizer' ),
        // biasConstraint : JoiEx.constraint().default( null ).description( 'Bias constraint' ),

        weightInitializer: JoiEx.initializer().default('random-uniform').description('Weight initializer'),

        // kernelRegularizer : JoiEx.regularizer().default( 'l2' ).description( 'Kernel regularizer' ),
        // kernelConstraint : JoiEx.constraint().default( 'max-norm' ).description( 'Kernel constraint' ),

        // activityRegularizer : JoiEx.regularizer().default( 'l1' ).description( 'Activity regularizer' )
      },
    );
  }
}
