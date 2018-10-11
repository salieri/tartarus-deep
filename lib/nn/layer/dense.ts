import { Layer, LayerDescriptor } from './layer';
/*import { Initializer } from '../initializer';
import { Regularizer } from '../regularizer';
import { Constraint } from '../constraint'; */
import { Activation } from '../activation';
import Joi from 'joi';


export class Dense extends Layer
{
	units( units : number ) : Dense
	{
		return <Dense>this.setParam( 'units', units );
	}


	activation( activation : string|Activation|null ) : Dense
	{
		return <Dense>this.setParam( 'activation', activation );
	}


	bias( bias : boolean ) : Dense
	{
		return <Dense>this.setParam( 'bias', bias );
	}


/*	biasInitializer( biasInitializer : Initializer|string|null  ) : Dense
	{
		return <Dense>this.setParam( 'biasInitializer', biasInitializer );
	}


	biasRegularizer( biasRegularizer : Regularizer|string|null  ) : Dense
	{
		return <Dense>this.setParam( 'biasRegularizer', biasRegularizer );
	}


	biasConstraint( biasConstraint : Constraint|string|null  ) : Dense
	{
		return <Dense>this.setParam( 'biasConstraint', biasConstraint );
	}


	kernelInitializer( kernelInitializer : Initializer|string|null  ) : Dense
	{
		return <Dense>this.setParam( 'kernelInitializer', kernelInitializer );
	}


	kernelRegularizer( kernelRegularizer : Regularizer|string|null  ) : Dense
	{
		return <Dense>this.setParam( 'kernelRegularizer', kernelRegularizer );
	}


	kernelConstraint( kernelConstraint : Constraint|string|null  ) : Dense
	{
		return <Dense>this.setParam( 'kernelConstraint', kernelConstraint );
	} */


	getDescriptor() : LayerDescriptor
	{
		return {
			units				: Joi.number().optional().default( 16 ).min( 1 ).description( 'Number of outputs' ),
			activation			: Joi.string().optional().default( 'identity' ).description( 'Activation function' ),

			bias				: Joi.boolean().optional().default( true ).description( 'Apply bias' ),
			//biasInitializer		: Joi.string().optional().default( 'zero' ).description( 'Bias initializer' ),
			//biasRegularizer		: Joi.string().optional().default( null ).description( 'Bias regularizer' ),
			//biasConstraint		: Joi.string().optional().default( null ).description( 'Bias constraint' ),

			//kernelInitializer	: Joi.string().optional().default( 'random-uniform' ).description( 'Kernel initializer' ),
			//kernelRegularizer	: Joi.string().optional().default( 'l2' ).description( 'Kernel regularizer' ),
			//kernelConstraint	: Joi.string().optional().default( 'max-norm' ).description( 'Kernel constraint' ),

			//activityRegularizer	: Joi.string().optional().default( 'l1' ).description( 'Activity regularizer' )
		};
	}
}
