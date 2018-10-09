import NDArray from '../ndarray';

interface NodeParams {
	[key: string]: any;
}


class Node
{
	public parent		: Node|null = null;
	public consumers	: Node[] = [];
	public params		: NodeParams;

	constructor( params : NodeParams = {} )
	{
		this.params = params;
	}

	calculate( x : NDArray ) : NDArray
	{
		return x;
	}

	public getDescriptor() : object
	{
		return this.params;
	}
}

export default Node;
