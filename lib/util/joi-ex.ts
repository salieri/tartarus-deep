import Joi from 'joi';

import {ClassManager} from './class-manager';

import * as activations from '../nn/activation';
import * as layers from '../nn/layer';
import * as losses from '../nn/loss';


function createCMExtension( name : string, cm : ClassManager ) : Function
{
	return ( joi: any ) => (
		{
			base: joi.any(),
			language : {
				coerceFailure : 'Unexpected data type passed to the coerce function'
			},
			name: name,
			coerce( value : any, state : any, options : any ) : any
			{
				try
				{
					return cm.coerce( value );
				}
				catch( e )
				{
					return (<any> this ).createError( `${name}.coerceFailure`, {}, state, options );
				}
			}
		}
	);
}


const customJoi = Joi.extend(
	[
		createCMExtension( 'activation', new ClassManager( activations, activations.Activation ) ),
		// createCMExtension( 'constraint', new ClassManager( constraints, constraints.Constraint ) ),
		// createCMExtension( 'initializer', new ClassManager( initializers, initializers.Initializer ) ),
		createCMExtension( 'layer', new ClassManager( layers, layers.Layer ) ),
		createCMExtension( 'loss', new ClassManager( losses, losses.Loss ) ),
		// createCMExtension( 'regularizer', new ClassManager( regularizers, regularizers.Regularizer ) )
	]
);

export {customJoi as JoiEx};


