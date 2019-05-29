import _ from 'lodash';

import { JoiEx, JoiExSchema } from '../../util';

import {
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


export enum ModelState {
  Created,
  Compiling,
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


export type RelaxedInputCollectionDefinition = number[]|number|NDArray|DeferredInputCollection|DeferredCollection;
export type RelaxedOutputCollectionDefinition = RelaxedInputCollectionDefinition;


export interface WeightCollection {
  [key: string]: number;
}


export interface MetricResultCollection {
  [key: string]: number;
}


export interface EvaluationResult {
  metrics: MetricResultCollection;
  loss: number;
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


  public constructor(params: ModelParamsInput = {}, name?: string) {
    super(params);

    this.session  = new Session(this.params.seed);

    Model.modelCounter += 1;

    this.name = this.validateName(name || `${this.constructor.name}#${Model.modelCounter}`);

    this.graph = new Graph(this.name);
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


  public setRawInputs(inputs: DeferredInputCollection): void {
    // this.rawInputs = inputs;
    this.graph.setRawInputs(inputs);
  }


  public getRawInputs(): DeferredInputCollection {
    return this.graph.getRawInputs();
  }


  public getRawOutputs(): DeferredInputCollection {
    return this.graph.getRawOutputs();
  }


  public getOutputNodes(): GraphNode[] {
    return this.graph.getOutputNodes();
  }


  public push(entity: GraphEntity): Model {
    const node = this.graph.push(entity);

    this.output(node);

    return this;
  }


  protected static coerceOutput(definition: RelaxedOutputCollectionDefinition): DeferredInputCollection {
    if (_.isNumber(definition) === true) {
      return Model.coerceInput(new NDArray([definition as number]));
    }

    if ((_.isArray(definition) === true) && (_.isNumber((definition as any[])[0]))) {
      return Model.coerceInput(new NDArray(definition as number[]));
    }

    return Model.coerceInput(definition);
  }


  protected static coerceInput(definition: RelaxedInputCollectionDefinition): DeferredInputCollection {
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


  public input(definition: RelaxedInputCollectionDefinition): Model {
    this.setRawInputs(Model.coerceInput(definition));

    return this;
  }


  public output(entities: EntityIdentifier|EntityIdentifier[]): Model {
    this.graph.setOutputNodes(entities);

    return this;
  }


  public getState(): ModelState {
    return this.state;
  }


  public async compile(): Promise<void> {
    if (this.state !== ModelState.Created) {
      throw new Error('Model has already been compiled');
    }

    this.state = ModelState.Compiling;

    await this.graph.compile();

    this.state = ModelState.Compiled;
  }


  public async fit(): Promise<void> {
    // moo
  }


  public async evaluate(input: RelaxedInputCollectionDefinition, expectedOutput: RelaxedOutputCollectionDefinition):
    Promise<EvaluationResult> {
    const output = await this.predict(input);
    const coercedExepctedOutput = Model.coerceOutput(expectedOutput);

    if (!_.isEqual(output.getKeys().sort(), coercedExepctedOutput.getKeys().sort())) {
      throw new Error(
        `Test labels for model '${this.getName()}' do not match with model output -- `
        + `(${output.getKeys()}) vs (${coercedExepctedOutput.getKeys()})`,
      );
    }

    const loss = this.calculateWeightedScore(output, coercedExepctedOutput, this.params.loss);
    const metrics = this.calculateMetrics(output, coercedExepctedOutput);

    return {
      metrics,
      loss,
    };
  }


  protected calculateMetrics(output: DeferredInputCollection, expectedOutput: DeferredInputCollection): MetricResultCollection {
    return _.zipObject(
      _.keys(this.params.metrics),
      _.map(
        this.params.metrics,
        (metric: Metric) => this.calculateWeightedScore(output, expectedOutput, metric),
      ),
    );
  }


  protected calculateWeightedScore(output: DeferredInputCollection, expectedOutput: DeferredInputCollection, loss: Loss|Metric): number {
    return _.reduce(
      output.getKeys(),
      (accumulator: number, key: string): number => {
        const outputValue = new Vector(output.get(key).getDefault().get());
        const expectedOuputValue = new Vector(expectedOutput.get(key).getDefault().get());

        return accumulator + this.getOutputWeight(key) * loss.calculate(outputValue, expectedOuputValue);
      },
      0,
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


  public async predict(input: RelaxedInputCollectionDefinition): Promise<DeferredInputCollection> {
    const preparedInput = Model.coerceInput(input);

    this.unsetOutputValues();

    this.graph.assign(preparedInput);

    await this.forward();

    return this.getRawOutputs();
  }


  public async forward(): Promise<void> {
    await this.graph.forward();
  }


  public async backward(): Promise<void> {
    await this.graph.backward();
  }


  public unsetOutputValues(): void {
    this.graph.unsetOutputValues();
  }
}

export default Model;
