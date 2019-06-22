import _ from 'lodash';

import {
  JoiEx,
  JoiExSchema,
  ContextLogger,
  Logger,
  MuteLogger, ConsoleLogger,
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
import { Parameterized, Parameters } from '../../generic';
import { DeferredCollection, DeferredInputCollection } from '../symbols';
import { Cost } from '../cost';
import { Loss } from '../loss';
import { Metric } from '../metric';
import { FitterParams, ModelFitter } from './fitter';
import { DeferredInputFeed } from '../../feed';
import { Optimizer } from '../optimizer';


export enum ModelState {
  Created,
  Compiled,
  Initialized,
}


export interface ModelParamsInput extends Parameters {
  seed?: string;
  cost?: Cost|string;
  loss?: Loss|string;
  metrics?: Metric[]|string[];
  optimizer?: Optimizer|string;
}


export interface ModelParamsCoerced extends ModelParamsInput {
  cost: Cost;
  loss: Loss;
  metrics: Metric[];
  optimizer: Optimizer;
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


export interface IterationResult {
  prediction: DeferredInputCollection;
  loss: DeferredInputCollection;
}


export class Model extends Parameterized<ModelParamsInput, ModelParamsCoerced> {
  private static modelCounter: number = 0;

  protected state: ModelState = ModelState.Created;

  protected graph: Graph;

  protected session: Session;

  protected readonly name: string;

  protected outputWeights: WeightCollection = {};

  protected evaluation?: EvaluationResult;

  protected prediction?: DeferredInputCollection;

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

    this.assignTrainingLabels(coercedExpectedOutput);

    await this.backward();
    await this.optimize();
  }


  public async fitBetter(
    params: FitterParams = {},
    data: DeferredInputFeed,
  ): Promise<void> {
    const fitter = new ModelFitter(this, data, params);

    await fitter.fit();
  }


  public assignTrainingLabels(labels: DeferredInputCollection): void {
    this.graph.assignTrainingLabels(labels);
  }


  public getRawTrainingLabels(): DeferredInputCollection {
    return new DeferredInputCollection(); // fake fake fake kludge kluge
  }


  public async iterate(
    input: DeferredInputCollection,
    expectedOutput: DeferredInputCollection,
  ): Promise<IterationResult> {
    const prediction = await this.predict(input);

    this.assignTrainingLabels(expectedOutput);

    await this.backward();

    const loss = this.calculateOutputScores(prediction, expectedOutput, this.params.loss);

    return {
      prediction,
      loss,
    };
  }


  public async evaluate(
    input: RelaxedDeclarationCollectionDefinition,
    expectedOutput: RelaxedDataCollectionDefinition,
  ): Promise<EvaluationResult> {
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


  public async predict(input: RelaxedDataCollectionDefinition): Promise<DeferredInputCollection> {
    delete this.prediction;
    delete this.evaluation;

    const preparedInput = Model.coerceData(input); // coerceOutput / RelaxedOutputCollectionDefinition is correct

    this.graph.unsetIterationValues();

    this.graph.assignInput(preparedInput);

    await this.forward();

    this.prediction = this.getRawOutputs().snapshot();

    return this.prediction;
  }


  public async forward(): Promise<void> {
    delete this.evaluation; // evaluation results are no longer applicable

    await this.graph.forward();
  }


  public async backward(): Promise<void> {
    await this.graph.backward(this.params.loss);
  }


  public async optimize(): Promise<void> {
    await this.graph.optimize(this.params.optimizer);
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


  public getParamSchema(): JoiExSchema {
    return JoiEx.object().keys(
      {
        seed: JoiEx.string().optional().default('tartarus-random-seed').min(2),
        cost: JoiEx.cost().optional().default('mean'),
        loss: JoiEx.loss().optional().default('mean-squared-error'),
        optimizer: JoiEx.optimizer().optional().default('stochastic'),
      },
    );
  }
}


export default Model;
