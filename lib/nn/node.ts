import NDArray from '../ndarray';

class Node
{
	public parent		: Node|null = null;
	public consumers	: Node[] = [];

	constructor()
	{
	}

	calculate( x : NDArray, params : object ) : NDArray
	{
		return x;
	}
}

export default Node;
