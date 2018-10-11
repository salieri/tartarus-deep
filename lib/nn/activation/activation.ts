import { NDArray } from '../../ndarray';


/**
 * Activation function `g` takes net input `z`
 * and outputs non-linear result `a` that determines
 * how active each layer in the input should be
 *
 * z = wx + b
 * a = g( z )
 */

export interface ActivationParams {
	[key: string]: any;
}

export interface ActivationDescriptor {
	[key: string]: any;
}



export class Activation
{
	protected params : ActivationParams;


	constructor( params : ActivationParams = {} )
	{
		this.params = params;
	}


	public calculate( z : NDArray ) : NDArray
	{
		throw new Error( 'Not implemented' );
	}


	public getDescriptor() : ActivationDescriptor
	{
		return {};
	}
}
