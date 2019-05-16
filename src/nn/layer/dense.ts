import { Layer, LayerParams } from './layer';
import { Activation } from '../activation';
import { JoiEx, JoiExSchema } from '../../util';
import { Matrix, NDArray, Vector } from '../../math';
import { Initializer } from '../initializer';


export interface DenseParamsInput extends LayerParams {
  units: number;
  activation?: Activation|string,
  bias?: boolean;
  biasInitializer?: Initializer|string,
  weightInitializer?: Initializer|string;
}

export interface DenseParamsCoerced extends DenseParamsInput {
  activation: Activation;
  biasInitializer: Initializer;
  weightInitializer: Initializer;
}


export class Dense extends Layer<DenseParamsInput, DenseParamsCoerced> {
  /**
   * dW[L] =
   */
  protected async backwardExec(): Promise<void> {
    // empty on purpose
  }


  protected async forwardExec(): Promise<void> {
    const linearOutput    = this.calculate(this.input.get());
    const activatedOutput = this.activate(linearOutput);

    this.cache.setValue('linear', linearOutput);
    this.cache.setValue('activated', activatedOutput);

    this.output.set(activatedOutput);
  }


  /**
   * Z = A_prev * W + b
   */
  protected calculate(input: NDArray): NDArray {
    const weight = this.optimizer.getValue('weight') as Matrix;

    let output = weight.vecmul(new Vector(input.flatten()));

    if (this.params.bias) {
      output = output.add(this.optimizer.getValue('bias') as Vector) as Vector;
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


  protected async compileExec(): Promise<void> {
    const inputUnits = this.input.size();
    const units = this.params.units;

    this.optimizer.declare('weight', [units, inputUnits]);
    this.output.declare([units, 1]);

    if (this.params.bias) {
      this.optimizer.declare('bias', [units, 1]);
    }

    this.cache.declare('linear', [units, 1]);
    this.cache.declare('activated', [units, 1]);
  }


  protected async initializeExec(): Promise<void> {
    const wInit = this.params.weightInitializer;
    const weight = this.optimizer.get('weight');

    weight.set(await wInit.initialize(new NDArray(...weight.getDims())));

    if (this.params.bias) {
      const bInit = this.params.biasInitializer;
      const bias = this.optimizer.get('bias');

      bias.set(await bInit.initialize(new NDArray(...bias.getDims())));
    }
  }


  public getParamSchema(): JoiExSchema {
    return JoiEx.object().keys(
      {
        units: JoiEx.number().default(16).min(1).description('Number of outputs'),
        activation: JoiEx.activation().default('identity').description('Activation function'),

        bias: JoiEx.boolean().default(true).description('Apply bias'),
        biasInitializer: JoiEx.initializer().layer(this).default('zero').description('Bias initializer'),

        // biasRegularizer : JoiEx.regularizer().default( null ).description( 'Bias regularizer' ),
        // biasConstraint : JoiEx.constraint().default( null ).description( 'Bias constraint' ),

        weightInitializer: JoiEx.initializer().layer(this).default('random-uniform').description('Weight initializer'),

        // kernelRegularizer : JoiEx.regularizer().default( 'l2' ).description( 'Kernel regularizer' ),
        // kernelConstraint : JoiEx.constraint().default( 'max-norm' ).description( 'Kernel constraint' ),

        // activityRegularizer : JoiEx.regularizer().default( 'l1' ).description( 'Activity regularizer' )
      },
    );
  }
}
