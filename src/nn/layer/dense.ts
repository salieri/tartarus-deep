import { Layer, LayerParams } from './layer';
import { Activation } from '../activation';
import { JoiEx, JoiExSchema } from '../../util';
import { Matrix, Vector } from '../../math';
import { Initializer } from '../initializer';
import { KeyNotFoundError } from '../../error';
import { Optimizer } from '../optimizer';
import { Loss } from '../loss';


export interface DenseParamsInput extends LayerParams {
  units: number;
  activation?: Activation|string;
  bias?: boolean;
  biasInitializer?: Initializer|string;
  biasOptimizer?: Optimizer|string;
  weightInitializer?: Initializer|string;
  weightOptimizer?: Optimizer|string;
  loss?: Loss|string;
}

export interface DenseParamsCoerced extends DenseParamsInput {
  activation: Activation;
  biasInitializer: Initializer;
  biasOptimizer: Optimizer;
  weightInitializer: Initializer;
  weightOptimizer: Optimizer;
  loss: Loss;
}


export class Dense extends Layer<DenseParamsInput, DenseParamsCoerced> {
  public static readonly WEIGHT_MATRIX = 'weight';

  public static readonly BIAS_VECTOR = 'bias';

  public static readonly LINEAR_OUTPUT = 'linear';

  public static readonly ACTIVATED_OUTPUT = 'activated';

  public static readonly WEIGHT_ERROR = 'weight-error';

  public static readonly BIAS_ERROR = 'bias-error';


  protected async optimizeExec(): Promise<void> {
    const fitter = this.data.fitter;
    const optimizer = this.data.optimizer;

    const weightError = fitter.getValue(Dense.WEIGHT_ERROR, Matrix);
    const weights = optimizer.getValue(Dense.WEIGHT_MATRIX, Matrix);
    const optimizedWeight = this.params.weightOptimizer.optimize(weights, weightError);

    optimizer.setValue(Dense.WEIGHT_MATRIX, optimizedWeight, Matrix);

    if (this.params.bias) {
      const biasError = fitter.getValue(Dense.BIAS_ERROR, Vector);
      const bias = optimizer.getValue(Dense.BIAS_VECTOR, Vector);
      const optimizedBias = this.params.biasOptimizer.optimize(bias, biasError);

      optimizer.setValue(Dense.BIAS_VECTOR, optimizedBias, Vector);
    }
  }


  // dActivated/dLinear = dO/dZ = dOut/dNet
  protected calculateActivationDerivative(): Vector {
    const output = this.data.output;
    const train = this.data.trainer;

    const activated = output.getValue(Dense.ACTIVATED_OUTPUT, Vector);
    const linear = output.getValue(Dense.LINEAR_OUTPUT, Vector);
    const y = train.hasDefaultValue() ? train.getDefaultValue(Vector) : undefined;

    return this.params.activation.derivative(activated, linear, y);
  }


  /**
   * @link https://www.youtube.com/watch?v=x_Eamf8MHwU
   */
  protected async backwardExec(): Promise<void> {
    const backpropOutput = this.data.backpropOutput;
    const fitter = this.data.fitter;

    // dError/dLinear
    const errorTerm = this.calculateErrorTerm();

    // dLinear/dWeights
    const weightError = this.calculateWeightDerivative(errorTerm);

    backpropOutput.setValue(Layer.ERROR_TERM, errorTerm, Vector);
    backpropOutput.setValue(Dense.WEIGHT_MATRIX, this.data.optimizer.getValue(Dense.WEIGHT_MATRIX, Matrix), Matrix);
    fitter.setValue(Dense.WEIGHT_ERROR, weightError, Matrix);

    if (this.params.bias) {
      // dLinear/dBias
      const biasError = this.calculateBiasDerivative(errorTerm);

      fitter.setValue(Dense.BIAS_ERROR, biasError, Vector);
    }
  }


  /**
   * dError/dActivated = L'(yHat, y)
   */
  protected calculateActivationErrorDerivativeFromLabel(): Vector {
    const yHat = this.data.output.getDefaultValue(Vector);
    const y = this.data.trainer.getDefaultValue(Vector);

    return this.params.loss.gradient(yHat, y);
  }


  /**
   * @link https://brilliant.org/wiki/backpropagation/
   * dError/dLinear = (dError/dActivated) * (dActivated/dLinear)
   */
  protected calculateErrorTermFromLabel(): Vector {
    // dError/dActivated
    const loss = this.calculateActivationErrorDerivativeFromLabel();

    // dActivated/dLinear
    const derivative = this.calculateActivationDerivative();

    // (a[final] - y) = (yHat - y) = -(y - yHat) = dErrorTotal / dOutput
    // layerError = g'(a[final])(yHat - y)
    // return derivative.mul(yHat.sub(y));

    // dError/dLinear = (dError/dActivated) * (dActivated/dLinear)
    return loss.mul(derivative);
  }


  /**
   * dError/dActivated = (dError/dLinearNext) . (dLinearNext/dActivated) = errorTermNext . weightNext
   */
  protected calculateActivationErrorDerivativeFromChain(): Vector {
    const backpropInput = this.data.backpropInput;

    // (dLinearNext/dActivated) = weights[l+1]
    const weightNext = backpropInput.getValue(Dense.WEIGHT_MATRIX, Matrix);

    // dError[l+1]/dLinear[l+1]
    const dErrorOverDLinearNext = backpropInput.getValue(Layer.ERROR_TERM, Vector);

    // dError/dActivated
    return weightNext.transpose().vecmul(dErrorOverDLinearNext);
  }


  /**
   * dError/dLinear[l] = (weights[l+1]T . (dError[l+1]/dLinear[l+1])) .* (dActivated/dLinear)
   * dError/dLinear = (dError/dActivated) * (dActivated/dLinear)
   */
  protected calculateErrorTermFromChain(): Vector {
    // dError/dActivated
    const dErrorOverDActivated = this.calculateActivationErrorDerivativeFromChain();

    // dActivated/dLinear
    const dActivatedOverDLinear = this.calculateActivationDerivative();

    // dError/dLinear
    return dErrorOverDActivated.mul(dActivatedOverDLinear);
  }


  /**
   * dError/dLinear
   */
  protected calculateErrorTerm(): Vector {
    return this.data.trainer.hasDefaultValue() ? this.calculateErrorTermFromLabel() : this.calculateErrorTermFromChain();
  }


  /**
   * dLinear/dWeights = (dError/dLinear) (o) X
   */
  protected calculateWeightDerivative(errorTerm: Vector): Matrix {
    const inputVector = this.data.input.getDefaultValue(Vector);

    return errorTerm.outer(inputVector); // inputVector.outer(errorTerm).transpose();
  }


  /**
   * dLinear/dBias = sum(dError/dLinear)
   */
  protected calculateBiasDerivative(errorTerm: Vector): Vector {
    return this.data.trainer.hasDefaultValue()
      ? new Vector([errorTerm.sum()])
      : new Vector([this.data.optimizer.getValue(Dense.BIAS_VECTOR, Vector).dot(errorTerm)]);
  }


  protected async forwardExec(): Promise<void> {
    const output = this.data.output;
    const linearOutput = this.calculate(this.data.input.getDefaultValue(Vector));
    const activatedOutput = this.activate(linearOutput);

    output.setValue(Dense.LINEAR_OUTPUT, linearOutput, Vector);
    output.setValue(Dense.ACTIVATED_OUTPUT, activatedOutput, Vector);
  }


  /**
   * Z = A_prev * W + b
   */
  protected calculate(input: Vector): Vector {
    const optimizer = this.data.optimizer;
    const weight = optimizer.getValue(Dense.WEIGHT_MATRIX, Matrix);

    let output = weight.vecmul(new Vector(input.flatten()));

    if (this.params.bias) {
      const bias = optimizer.getValue(Dense.BIAS_VECTOR, Vector).getAt([0]);

      output = output.add(bias);
    }

    return output;
  }


  /**
   * A = g(Z)
   */
  protected activate(linearOutput: Vector): Vector {
    const activation = this.params.activation;

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
    this.raw.trainingLabels.setDefault(this.data.trainer);
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
    const train = this.data.trainer;
    const backpropInput = this.data.backpropInput;
    const backpropOutput = this.data.backpropOutput;
    const fitter = this.data.fitter;
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

      backpropInput.require(Layer.ERROR_TERM);
      backpropInput.require(Dense.WEIGHT_MATRIX);
    }

    const weightDims = optimizer.get(Dense.WEIGHT_MATRIX).getDims();

    backpropOutput.declare(Layer.ERROR_TERM, this.countOutputUnits());
    backpropOutput.declare(Dense.WEIGHT_MATRIX, weightDims);
    fitter.declare(Dense.WEIGHT_ERROR, weightDims);

    if (this.params.bias) {
      fitter.declare(Dense.BIAS_ERROR, 1);
    }
  }


  protected async initializeExec(): Promise<void> {
    const optimizer = this.data.optimizer;
    const wInit = this.params.weightInitializer;
    const weight = optimizer.get(Dense.WEIGHT_MATRIX);

    weight.set(await wInit.initialize(new Matrix(...weight.getDims())), Matrix);

    if (this.params.bias) {
      const bInit = this.params.biasInitializer;
      const bias = optimizer.get(Dense.BIAS_VECTOR);

      bias.set(await bInit.initialize(new Vector(...bias.getDims())), Vector);
    }
  }


  public getParamSchema(): JoiExSchema {
    return JoiEx.object().keys(
      {
        units: JoiEx.number().default(16).min(1).description('Number of outputs'),
        activation: JoiEx.activation().default('identity').description('Activation function'),

        bias: JoiEx.boolean().default(true).description('Apply bias'),
        biasInitializer: JoiEx.initializer().default('zero').description('Bias initializer'),
        biasOptimizer: JoiEx.optimizer().default('stochastic').description('Bias optimizer'),

        // biasRegularizer : JoiEx.regularizer().default( null ).description( 'Bias regularizer' ),
        // biasConstraint : JoiEx.constraint().default( null ).description( 'Bias constraint' ),

        weightInitializer: JoiEx.initializer().default('random-uniform').description('Weight initializer'),
        weightOptimizer: JoiEx.optimizer().default('stochastic').description('Weight optimizer'),

        loss: JoiEx.loss().default('mean-squared-error').description('Loss function'),

        // kernelRegularizer : JoiEx.regularizer().default( 'l2' ).description( 'Kernel regularizer' ),
        // kernelConstraint : JoiEx.constraint().default( 'max-norm' ).description( 'Kernel constraint' ),

        // activityRegularizer : JoiEx.regularizer().default( 'l1' ).description( 'Activity regularizer' )
      },
    );
  }
}
