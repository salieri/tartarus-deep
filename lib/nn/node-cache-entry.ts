import _ from 'lodash';
import Node from './node';

class NodeCacheEntry
{
	node		: Node;
	inputs		: NodeCacheEntry[] = [];
	outputs		: NodeCacheEntry[] = [];
	connected	: boolean = false;
	layer		: number = 0;

	constructor( node : Node )
	{
		this.node = node;
	}


	addOutput( entry : NodeCacheEntry )
	{
		this.outputs.push( entry );
	}


	addInput( entry : NodeCacheEntry )
	{
		this.inputs.push( entry );
	}


	removeOutput( entry : NodeCacheEntry )
	{
		_.remove( this.inputs, ( input : NodeCacheEntry ) => ( entry === input ) );
	}


	removeInput( entry : NodeCacheEntry )
	{
		_.remove( this.outputs, ( output : NodeCacheEntry ) => ( entry === output ) );
	}
}


export default NodeCacheEntry;
