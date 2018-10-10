import {Layer, LayerDescriptor} from '.';
import Joi from 'joi';


export class Dense extends Layer
{
	units( units : number ) : Dense
	{
		return <Dense>this.setParam( 'units', units );
	}


	activation( activation : string ) : Dense
	{
		return <Dense>this.setParam( 'activation', activation );
	}


	bias( bias : boolean ) : Dense
	{
		return <Dense>this.setParam( 'bias', bias );
	}


	initializer( initializer : object ) : Dense
	{
		return <Dense>this.setParam( 'initializer', initializer );
	}


	regularizer( regularizer : object ) : Dense
	{
		return <Dense>this.setParam( 'regularizer', regularizer );
	}


	constraint( constraint : object ) : Dense
	{
		return <Dense>this.setParam( 'constraint', constraint );
	}


	getDescriptorCoercers() : DescriptorCoercer
	{
		return {
			'activation'			: 'activation',
			'initializer.bias'		: 'initializer',
			'initializer.kernel'	: 'initializer',
			'regularizer.bias'		: 'regularizer',
			'regularizer.kernel'	: 'regularizer',
			'regularizer.activity'	: 'regularizer',
			'constraint.bias'		: 'constraint',
			'constraint.kernel'		: 'constraint'
		}
	}


	getDescriptor() : LayerDescriptor
	{
		return {
			units		: Joi.number().optional().default( 16 ).min( 1 ).description( 'Number of outputs' ),
			activation	: Joi.string().optional().default( 'identity' ).description( 'Activation function' ),
			bias		: Joi.boolean().optional().default( true ).description( 'Apply bias' ),
			initializer	: Joi.object(
				{
					bias	: Joi.string().optional().default( 'zero' ).description( 'Bias initializer' ),
					kernel	: Joi.string().optional().default( 'random-uniform' ).description( 'Kernel initializer' )
				}
				).optional().default( { bias : 'zero', kernel : 'random-uniform' } ).description( 'Initializers' ),
			regularizer	: Joi.object(
				{
					bias	: Joi.string().optional().default( null ).description( 'Bias regularizer' ),
					kernel	: Joi.string().optional().default( 'l2' ).description( 'Kernel regularizer' ),
					activity	: Joi.string().optional().default( 'l1' ).description( 'Activity regularizer' )
				}
			).optional().default( { bias : null, kernel: 'l2', activity : 'l1' } ).description( 'Regularizers' ),
			constraint : Joi.object(
				{
					bias	: Joi.string().optional().default( null ).description( 'Bias constraint' ),
					kernel	: Joi.string().optional().default( 'max-norm' ).description( 'Kernel constraint' )
				}
			).optional().default( { bias : null, kernel : 'max-norm' } ).description( 'Constraints' )
		};
	}
}
