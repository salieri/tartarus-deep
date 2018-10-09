import Node from '../node';
import NDArray from '../../ndarray';

/**
 * Activation function `g` takes net input `z`
 * and outputs non-linear result `a` that determines
 * how active each neuron in the input should be
 *
 * z = wx + b
 * a = g( z )
 */


class Activation extends Node
{
	constructor()
	{
		super();
	}

	public calculate( z : NDArray ) : NDArray
	{
		throw new Error( 'Not implemented' );
	}
}


export default Activation;
