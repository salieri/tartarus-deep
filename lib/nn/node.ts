import NDArray from '../ndarray';

class Node
{
	public parent		: Node|null = null;
	public consumers	: Node[] = [];
	public params		: object;

	constructor( params : object = {} )
	{
		this.params = params;
	}

	calculate( x : NDArray, params : object ) : NDArray
	{
		return x;
	}

	public getDescriptor() : object
	{
		return this.params;
	}
}

export default Node;
