import NDArray from '../ndarray';
import Model from './model';

interface NodeParams {
	[key: string]: any;
}


class Node
{
	public params	: NodeParams;
	public model	: Model;

	constructor( model : Model, params : NodeParams = {} )
	{
		this.model	= model;
		this.params	= params;
	}


	calculate( x : NDArray ) : NDArray
	{
		return x;
	}


	public getDescriptor() : NodeParams
	{
		return {};
	}


	public forward( input : NDArray )
	{
	}


	public backward( output : NDArray )
	{
	}


	public getLayerVar( name : String ) : any
	{
	}


	public setLayerVar( name : String, value : any )
	{
	}
}

export default Node;
