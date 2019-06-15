import { Layer, LayerParams } from './layer';
import { Activation } from '../activation';
import { JoiEx, JoiExSchema } from '../../util';
import { Matrix, NDArray, Vector } from '../../math';
import { Initializer } from '../initializer';
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
  public static readonly WEIGHT_MATRIX = 'weight';

  public static readonly BIAS_VECTOR = 'bias';

  public static readonly LINEAR_OUTPUT = 'linear';

  public static readonly ACTIVATED_OUTPUT = 'activated';


  protected derivative(): NDArray {
    const output = this.data.output;
    const train = this.data.train;

    const activated = output.getValue(Dense.ACTIVATED_OUTPUT);
    const linear = output.getValue(Dense.LINEAR_OUTPUT);
    const y = train.hasDefaultValue() ? train.getDefaultValue() : undefined;

    return this.params.activation.derivative(activated, linear, y);
  }


  /**
   * @link https://www.youtube.com/watch?v=x_Eamf8MHwU
   */
  protected async backwardExec(): Promise<void> {
    this.calculateBackward();
  }


  /**
   * @link https://brilliant.org/wiki/backpropagation/
   */
  protected calculateErrorTermFromLabel(): Vector {
    const yHat = new Vector(this.data.output.getDefaultValue());
    const y = new Vector(this.data.train.getDefaultValue());

    // (a[final] - y) = (yHat - y) = -(y - yHat) = dErrorTotal / dOutput
    // layerError = g'(a[final])(yHat - y)
    return new Vector(this.derivative().mul(yHat.sub(y)));
  }


  protected calculateErrorTermFromChain(): Vector {
    const backpropInput = this.data.backpropInput;
    const layerErrorNext = new Vector(backpropInput.getValue(Layer.DERIVATIVE));
    const weightNext = new Matrix(backpropInput.getValue(Dense.WEIGHT_MATRIX));

    // layerError = (wNext)T dNext .* g'(z)
    return new Vector(weightNext.transpose().vecmul(layerErrorNext).mul(this.derivative()));
  }


  protected calculateBackward(): void {
    const backpropOutput = this.data.backpropOutput;
    const errorTerm = this.data.train.hasDefaultValue() ? this.calculateErrorTermFromLabel() : this.calculateErrorTermFromChain();

    backpropOutput.setValue(Layer.DERIVATIVE, errorTerm);
    backpropOutput.setValue(Dense.WEIGHT_MATRIX, this.data.optimizer.getValue(Dense.WEIGHT_MATRIX));
  }


  protected calculateWeightDerivative(errorTerm: Vector): NDArray {
    const inputVector = new Vector(this.data.input.getDefaultValue());

    return inputVector.outer(errorTerm);
  }


  protected calculateBiasDerivative(errorTerm: Vector): number {
    return errorTerm.sum();
  }


  protected async forwardExec(): Promise<void> {
    const output = this.data.output;
    const linearOutput = this.calculate(this.data.input.getDefault().get());
    const activatedOutput = this.activate(linearOutput);

    output.setValue(Dense.LINEAR_OUTPUT, linearOutput);
    output.setValue(Dense.ACTIVATED_OUTPUT, activatedOutput);
  }


  /**
   * Z = A_prev * W + b
   */
  protected calculate(input: NDArray): NDArray {
    const optimizer = this.data.optimizer;
    const weight = new Matrix(optimizer.getValue(Dense.WEIGHT_MATRIX));

    let output = weight.vecmul(new Vector(input.flatten()));

    if (this.params.bias) {
      const bias = optimizer.getValue(Dense.BIAS_VECTOR).getAt([0]);

      output = output.add(bias) as Vector;
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


  protected resolveBackpropInput(): void {
    const backpropInput = this.data.backpropInput;
    const rawBackpropInputs = this.raw.backpropInputs;

    // This needs rewriting to deal with cases where a layer has multiple outputs
    // This needs rewriting to deal with bias, not just weight
    try {
      backpropInput.setCollection(rawBackpropInputs.getDefault());
      return;
    } catch (err) {
      if (!(err instanceof KeyNotFoundError)) {
        throw err;
      }
    }

    if (rawBackpropInputs.count() < 1) {
      throw new Error(`Missing input for dense layer '${this.getName()}'`);
    }

    if (rawBackpropInputs.count() > 1) {
      // throw new Error(`Too many inputs for a dense layer '${this.getName()}'`);
      // DO SOMETHING
    }

    backpropInput.setCollection(rawBackpropInputs.first());
  }


  protected resolveInput(): void {
    const input = this.data.input;
    const rawInputs = this.raw.inputs;

    try {
      const defaultInput = rawInputs.getDefault();

      input.setCollection(defaultInput);
      return;
    } catch (err) {
      if (!(err instanceof KeyNotFoundError)) {
        throw err;
      }
    }

    if (rawInputs.count() < 1) {
      throw new Error(`Missing input for dense layer '${this.getName()}'`);
    }

    if (rawInputs.count() > 1) {
      throw new Error(`Too many inputs for a dense layer '${this.getName()}'`);
    }

    input.setCollection(rawInputs.first());
  }


  protected countOutputUnits(): number {
    return this.params.units;
  }


  protected countInputUnits(): number {
    return this.data.input.getDefault().countElements();
  }


  protected async compileInitialization(): Promise<void> {
    this.raw.trainingLabels.setDefault(this.data.train);
    this.raw.backpropOutputs.setDefault(this.data.backpropOutput);
    this.raw.outputs.setDefault(this.data.output);
  }


  protected async compileForwardPropagation(): Promise<void> {
    this.resolveInput();

    this.data.input.requireDefault();

    const optimizer = this.data.optimizer;
    const output = this.data.output;

    const inputUnits = this.countInputUnits();
    const units = this.countOutputUnits();

    optimizer.declare(Dense.WEIGHT_MATRIX, [units, inputUnits]);

    if (this.params.bias) {
      optimizer.declare(Dense.BIAS_VECTOR, 1);
    }

    output.declare(Dense.LINEAR_OUTPUT, [units]);
    output.declare(Dense.ACTIVATED_OUTPUT, [units]);

    output.setDefaultKey(Dense.ACTIVATED_OUTPUT);
  }


  protected async compileBackPropagation(): Promise<void> {
    const train = this.data.train;
    const backpropInput = this.data.backpropInput;
    const backpropOutput = this.data.backpropOutput;
    const optimizer = this.data.optimizer;

    const rawTrainingLabels = this.raw.trainingLabels;
    const rawBackpropInputs = this.raw.backpropInputs;

    train.setCollection(rawTrainingLabels.getDefault());

    if (rawBackpropInputs.count() === 0) {
      const trainColl  = train.getCollection();

      trainColl.declare(Layer.TRAINING_LABEL, this.countOutputUnits());
      trainColl.setDefaultKey(Layer.TRAINING_LABEL);
    } else {
      this.resolveBackpropInput();

      backpropInput.require(Layer.DERIVATIVE);
      backpropInput.require(Dense.WEIGHT_MATRIX);
    }

    backpropOutput.declare(Layer.DERIVATIVE, this.countOutputUnits());
    backpropOutput.declare(Dense.WEIGHT_MATRIX, optimizer.get(Dense.WEIGHT_MATRIX).getDims());
  }


  protected async initializeExec(): Promise<void> {
    const optimizer = this.data.optimizer;
    const wInit = this.params.weightInitializer;
    const weight = optimizer.get(Dense.WEIGHT_MATRIX);

    weight.set(await wInit.initialize(new NDArray(...weight.getDims())));

    if (this.params.bias) {
      const bInit = this.params.biasInitializer;
      const bias = optimizer.get(Dense.BIAS_VECTOR);

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
