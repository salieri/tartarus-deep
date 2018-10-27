import { Layer, LayerDescriptor } from './layer';
import { Activation } from '../activation';
import { JoiEx } from '../../util';
import { Vector, Matrix } from '../../math';

/*import { Initializer } from '../initializer';
import { Regularizer } from '../regularizer';
import { Constraint } from '../constraint'; */


export class Dense extends Layer
{
	units( units : number ) : Dense
	{
		return <Dense>this.setParam( 'units', units );
	}


	activation( activation : string|Activation ) : Dense
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


	getParameterDimensions()
	{

	}


	calculate( input : Vector ) : Vector
	{
		const weight	= <Matrix>this.session.get( 'weight' ),
			output		= weight.vecmul( input );

		if( this.params.bias )
		{
			return <Vector>output.add( <Vector>this.session.get( 'bias' ) );
		}

		return output;
	}


	compile()
	{
		this.session.register( 'weight', new Matrix( this.params.units, inputNodes ) );

		if( this.params.bias === true )
		{
			this.session.register( 'bias', new Vector( this.params.units ) );
		}
	}


	initialize()
	{
		this.register( 'weight', this.params.weightInitializer.initialize( this.get( 'weight' ) ) );

		if( this.params.bias === true )
		{
			this.register( 'bias', this.params.biasInitializer.initialize( this.get( 'bias' ) ) );
		}
	}


	getDescriptor() : LayerDescriptor
	{
		return {
			units				: JoiEx.number().default( 16 ).min( 1 ).description( 'Number of outputs' ),
			activation			: JoiEx.activation().default( 'identity' ).description( 'Activation function' ),

			bias				: JoiEx.boolean().default( true ).description( 'Apply bias' ),
			biasInitializer		: JoiEx.initializer().default( 'zero' ).description( 'Bias initializer' ),

			//biasRegularizer		: JoiEx.regularizer().default( null ).description( 'Bias regularizer' ),
			//biasConstraint		: JoiEx.constraint().default( null ).description( 'Bias constraint' ),

			weightInitializer	: JoiEx.initializer().default( 'random-uniform' ).description( 'Weight initializer' ),

			//kernelRegularizer	: JoiEx.regularizer().default( 'l2' ).description( 'Kernel regularizer' ),
			//kernelConstraint	: JoiEx.constraint().default( 'max-norm' ).description( 'Kernel constraint' ),

			//activityRegularizer	: JoiEx.regularizer().default( 'l1' ).description( 'Activity regularizer' )
		};
	}
}
