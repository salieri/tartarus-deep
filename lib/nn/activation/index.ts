import Node from '../node';
import NDArray from '../../ndarray';
import Joi from 'joi';


/**
 * Activation function `g` takes net input `z`
 * and outputs non-linear result `a` that determines
 * how active each neuron in the input should be
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



class Activation
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


export default Activation;
