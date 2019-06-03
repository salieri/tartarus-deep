import _ from 'lodash';

import {
  JoiEx,
  JoiExSchema,
  ContextLogger,
  Logger,
  MuteLogger,
} from '../../util';

import {
  CompilationStage,
  EntityIdentifier,
  Graph,
  GraphEntity,
  GraphNode,
} from '../graph';

import { Session } from '../session';
import { NDArray, Randomizer, Vector } from '../../math';
import { Parameterized } from '../../generic';
import { DeferredCollection, DeferredInputCollection } from '../symbols';
import { Cost } from '../cost';
import { Loss } from '../loss';
import { Metric } from '../metric';
import { Layer } from '../layer';


export enum ModelState {
  Created,
  Compiled,
  Initialized,
}


export interface ModelParamsInput {
  seed?: string;
  cost?: Cost|string;
  loss?: Loss|string;
  metrics?: Metric[]|string[];
}


export interface ModelParamsCoerced extends ModelParamsInput {
  cost: Cost;
  loss: Loss;
  metrics: Metric[];
}


export type RelaxedDeclarationCollectionDefinition = number[]|number|NDArray|DeferredInputCollection|DeferredCollection;
export type RelaxedDataCollectionDefinition = RelaxedDeclarationCollectionDefinition;


export interface WeightCollection {
  [key: string]: number;
}


export interface MetricResultCollection {
  [key: string]: DeferredInputCollection;
}


export interface EvaluationResult {
  metrics: MetricResultCollection;
  losses: DeferredInputCollection;
}


export class Model
  extends Parameterized<ModelParamsInput, ModelParamsCoerced>
  implements GraphEntity {
  private static modelCounter: number = 0;

  protected state: ModelState = ModelState.Created;

  protected graph: Graph;

  protected session: Session;

  protected readonly name: string;

  protected outputWeights: WeightCollection = {};

  protected evaluation?: EvaluationResult;

  protected logger: Logger = new MuteLogger();


  public constructor(params: ModelParamsInput = {}, name?: string) {
    super(params);

    Model.modelCounter += 1;

    this.name = this.validateName(name || `${this.constructor.name}#${Model.modelCounter}`);
    this.session = new Session(this.params.seed);
    this.graph = new Graph(this.name, this.session);

    this.setSession(this.session);
    this.setLogger(this.session.getLogger());
  }


  protected validateName(name: string): string {
    if (name.match(/[. ]/)) {
      throw new Error('Model names may not contain spaces or periods');
    }

    return name;
  }


  public getParamSchema(): JoiExSchema {
    return JoiEx.object().keys(
      {
        seed: JoiEx.string().optional().default('tartarus-random-seed').min(2),
        cost: JoiEx.cost().optional().default('mean'),
        loss: JoiEx.loss().optional().default('mean-squared-error'),
      },
    );
  }


  public getRandomizer(): Randomizer {
    return this.session.getRandomizer();
  }


  public getSession(): Session {
    return this.session;
  }


  public getName(): string {
    return this.name;
  }


  public add(entity: GraphEntity, parentEntities?: EntityIdentifier|EntityIdentifier[]): Model {
    this.graph.add(entity, parentEntities);

    return this;
  }


  public link(output: EntityIdentifier, input: EntityIdentifier): Model {
    this.graph.link(output, input);

    return this;
  }


  public getRawBackpropInputs(): DeferredInputCollection {
    return this.graph.getRawBackpropInputs();
  }


  public setRawBackpropInputs(inputs: DeferredInputCollection): void {
    this.graph.setRawBackpropInputs(inputs);
  }


  public setRawInputs(inputs: DeferredInputCollection): void {
    this.graph.setRawInputs(inputs);
  }


  public getRawInputs(): DeferredInputCollection {
    return this.graph.getRawInputs();
  }


  public getRawOutputs(): DeferredInputCollection {
    return this.graph.getRawOutputs();
  }


  public getRawBackpropOutputs(): DeferredInputCollection {
    return this.graph.getRawBackpropOutputs();
  }


  public getOutputNodes(): GraphNode[] {
    return this.graph.getOutputNodes();
  }


  public push(entity: GraphEntity): Model {
    const node = this.graph.push(entity);

    this.output(node);

    return this;
  }


  public static coerceData(definition: RelaxedDataCollectionDefinition): DeferredInputCollection {
    if (_.isNumber(definition) === true) {
      return Model.coerceDeclaration(new NDArray([definition as number]));
    }

    if ((_.isArray(definition) === true) && (_.isNumber((definition as any[])[0]))) {
      return Model.coerceDeclaration(new NDArray(definition as number[]));
    }

    return Model.coerceDeclaration(definition);
  }


  public static coerceDeclaration(definition: RelaxedDeclarationCollectionDefinition): DeferredInputCollection {
    if (definition instanceof  DeferredInputCollection) {
      return definition;
    }

    if (definition instanceof DeferredCollection) {
      return new DeferredInputCollection(definition);
    }

    if (definition instanceof NDArray) {
      return new DeferredInputCollection(new DeferredCollection(definition));
    }

    const dims = _.castArray(definition);
    const collection = new DeferredCollection();

    collection.declareDefault(dims);

    return new DeferredInputCollection(collection);
  }


  public input(definition: RelaxedDeclarationCollectionDefinition): Model {
    this.setRawInputs(Model.coerceDeclaration(definition));

    return this;
  }


  public output(entities: EntityIdentifier|EntityIdentifier[]): Model {
    this.graph.setOutputNodes(entities);

    return this;
  }


  public getState(): ModelState {
    return this.state;
  }


  protected async compileAsMember(stage: CompilationStage): Promise<void> {
    this.logger.debug('model.compile.stage', () => ({ stage: CompilationStage[stage], member: true }));

    await this.graph.compile(stage);

    if (stage === CompilationStage.Finalize) {
      this.state = ModelState.Compiled;
    }
  }


  protected async compileAsMaster(): Promise<void> {
    // eslint-disable-next-line guard-for-in, no-restricted-syntax
    for (const stage in _.filter(CompilationStage, _.isNumber)) {
      // eslint-disable-next-line no-await-in-loop
      await this.graph.compile(Number(stage));

      this.logger.debug('model.compile.stage', { stage: CompilationStage[stage] });
    }
  }


  public async compile(stage?: CompilationStage): Promise<void> {
    this.logger.info('model.compile');

    if (this.state !== ModelState.Created) {
      throw new Error('Model has already been compiled');
    }

    await (_.isUndefined(stage) ? this.compileAsMaster() : this.compileAsMember(stage));

    this.state = ModelState.Compiled;
  }


  public async initialize(): Promise<void> {
    this.logger.info('model.initialize');

    if (this.state !== ModelState.Compiled) {
      throw new Error('Model has to be compiled before it can be initialized');
    }

    await this.graph.initialize();

    this.state = ModelState.Initialized;
  }


  public async fit(input: RelaxedDeclarationCollectionDefinition, expectedOutput: RelaxedDataCollectionDefinition): Promise<void> {
    await this.evaluate(input, expectedOutput);

    const coercedExpectedOutput = Model.coerceData(expectedOutput);
    const outputDerivatives = this.calculateTopLevelBackpropInputs(coercedExpectedOutput);

    this.graph.unsetBackpropInputValues();
    this.graph.unsetBackpropOutputValues();
    this.graph.assignBackpropInput(outputDerivatives);

    await this.backward();
  }


  protected calculateTopLevelBackpropInputs(expectedOutput: DeferredInputCollection): DeferredInputCollection {
    const rawOutputs = this.getRawOutputs();
    const backpropInputs = this.getRawBackpropInputs();

    if (!this.evaluation) {
      throw new Error('Derivatives cannot be calculated before forward propagation');
    }

    const evaluation = this.evaluation;

    _.each(
      backpropInputs.getKeys(),
      (key: string) => {
        const yHat = rawOutputs.get(key).getDefault().get();
        const y = expectedOutput.get(key).getDefault().get();

        const dETotalOverOutput = y.sub(yHat).neg();

        const coll = backpropInputs.get(key).getCollection();

        coll.setValue(Layer.DERIVATIVE, dETotalOverOutput);
        coll.setValue(Layer.LOSS, evaluation.losses.get(key).getDefault().get());

        backpropInputs.set(key, coll);
      },
    );

    return backpropInputs;
  }


  public async evaluate(input: RelaxedDeclarationCollectionDefinition, expectedOutput: RelaxedDataCollectionDefinition):
    Promise<EvaluationResult> {
    const output = await this.predict(input);
    const coercedExpectedOutput = Model.coerceData(expectedOutput);

    if (!_.isEqual(output.getKeys().sort(), coercedExpectedOutput.getKeys().sort())) {
      throw new Error(
        `Test labels for model '${this.getName()}' do not match with model output -- `
        + `(${output.getKeys()}) vs (${coercedExpectedOutput.getKeys()})`,
      );
    }

    const losses = this.calculateOutputScores(output, coercedExpectedOutput, this.params.loss);
    const metrics = this.calculateMetrics(output, coercedExpectedOutput);

    this.evaluation = {
      losses,
      metrics,
    };

    return this.evaluation;
  }


  protected calculateOutputScores(
    output: DeferredInputCollection,
    expectedOutput: DeferredInputCollection,
    loss: Loss|Metric,
  ): DeferredInputCollection {
    const result = new DeferredInputCollection();

    _.each(
      output.getKeys(),
      (key: string) => {
        const outputValue = new Vector(output.get(key).getDefault().get());
        const expectedOuputValue = new Vector(expectedOutput.get(key).getDefault().get());

        const lossScore = this.getOutputWeight(key) * loss.calculate(outputValue, expectedOuputValue);

        result.set(key, new DeferredCollection(new NDArray([lossScore])));
      },
    );

    return result;
  }


  protected calculateMetrics(output: DeferredInputCollection, expectedOutput: DeferredInputCollection): MetricResultCollection {
    return _.zipObject(
      _.keys(this.params.metrics),
      _.map(
        this.params.metrics,
        (metric: Metric) => this.calculateOutputScores(output, expectedOutput, metric),
      ),
    );
  }


  protected getOutputWeight(key: string): number {
    if (key in this.outputWeights) {
      return this.outputWeights[key];
    }

    return 1.0;
  }


  // protected static castInputArray(inputs: RelaxedInputCollectionDefinition|RelaxedInputCollectionDefinition[]):
  //   RelaxedInputCollectionDefinition[] {
  //   if (
  //     (_.isArray(inputs))
  //     && (
  //       (inputs[0] instanceof NDArray)
  //       || (inputs[0] instanceof DeferredInputCollection)
  //       || (inputs[0] instanceof DeferredCollection)
  //     )
  //   ) {
  //     return _.map(
  //       inputs,
  //       (input: RelaxedInputCollectionDefinition) => (Model.prepareInputCollection(input)),
  //     );
  //   }
  //
  //   return [Model.prepareInputCollection(inputs as RelaxedInputCollectionDefinition)];
  // }


  public async predict(input: RelaxedDataCollectionDefinition): Promise<DeferredInputCollection> {
    const preparedInput = Model.coerceData(input); // coerceOutput / RelaxedOutputCollectionDefinition is correct

    this.graph.unsetOutputValues();
    this.graph.unsetInputValues();

    this.graph.assignInput(preparedInput);

    await this.forward();

    return this.getRawOutputs().snapshot();
  }


  public async forward(): Promise<void> {
    delete this.evaluation; // evaluation results are no longer applicable

    await this.graph.forward();
  }


  public async backward(): Promise<void> {
    await this.graph.backward();
  }


  public unsetOutputValues(): void {
    this.graph.unsetOutputValues();
  }


  public unsetInputValues(): void {
    this.graph.unsetInputValues();
  }


  public unsetBackpropOutputValues(): void {
    this.graph.unsetBackpropOutputValues();
  }


  public unsetBackpropInputValues(): void {
    this.graph.unsetBackpropInputValues();
  }


  public setSession(session: Session): void {
    this.session = session;

    this.graph.setSession(this.session);
  }


  public setLogger(parentLogger: Logger): void {
    this.logger = new ContextLogger(parentLogger, this.getName());

    this.graph.setLogger(this.logger);
  }


  public getGraph() : Graph {
    return this.graph;
  }
}

export default Model;
