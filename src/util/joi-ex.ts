import Joi from 'joi';
import _ from 'lodash';

import { ClassManager } from './class-manager';

import * as activations from '../nn/activation';
// import * as layers from '../nn/layer'; // -- this will cause circular dependencies
import * as losses from '../nn/loss';
import * as initializers from '../nn/initializer';
import * as randomizers from '../math/randomizer';


function createCMExtension(name: string, cm: ClassManager): Function {
  return (joi: any) => (
    {
      name,
      base: joi.any(),
      language: {
        coerceFailure: 'Unexpected data type passed to the coerce function'
      },
      coerce(value: any, state: any, options: any): any {
        try {
          return cm.coerce(value || _.get(this, '_flags.default'));
        } catch (e) {
          return (<any>this).createError(`${name}.coerceFailure`, {}, state, options);
        }
      }
    }
  );
}


const customJoi = Joi.extend(
  [
    createCMExtension('activation', new ClassManager(activations, activations.Activation)),
    // createCMExtension( 'constraint', new ClassManager( constraints, constraints.Constraint ) ),
    createCMExtension('initializer', new ClassManager(initializers, initializers.Initializer)),

    // createCMExtension( 'layer', new ClassManager( layers, layers.Layer ) ), // -- this will cause circular dependencies

    createCMExtension('loss', new ClassManager(losses, losses.Loss)),
    createCMExtension('randomizer', new ClassManager(randomizers, randomizers.Randomizer))

    // createCMExtension( 'regularizer', new ClassManager( regularizers, regularizers.Regularizer ) )
  ]
);

export { customJoi as JoiEx };

