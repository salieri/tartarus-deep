import { Model } from '../model';

// eslint-disable-next-line
export interface ModelFactoryOpts {}


export class ModelFactory {

  public static factory(opts: ModelFactoryOpts): Model {
    throw new Error('Not implemented');
  }

}
