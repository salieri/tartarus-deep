import _ from 'lodash';
import Node from './node';
import NodeCacheEntry from './node-cache-entry';


class NodeCache
{
	protected nodes		: NodeCacheEntry[] = [];
	protected compiled	: boolean = false;


	public add( node : Node, parentNode? : Node )
	{
		this.canModify();

		if( this.exists( node ) === true )
		{
			throw new Error( `Node '${node.name}' already exists in this model` );
		}

		if( parentNode )
		{
			if( this.exists( parentNode ) === false )
			{
				throw new Error( `Parent node '${node.name}' does not exist in this model` );
			}

			if( parentNode === node )
			{
				throw new Error( `Parent node cannot be same instance than the node being added` );
			}
		}

		const entry = new NodeCacheEntry( node );

		this.nodes.push( entry );

		if( parentNode )
		{
			this.link( parentNode, node );
		}
	}


	public link( outputNode : Node, inputNode : Node )
	{
		this.canModify();

		const outputNodeEntry	= this.find( outputNode ),
			inputNodeEntry		= this.find( inputNode );

		this.checkForCircularLinks( outputNodeEntry, 'backward', inputNodeEntry );
		this.checkForCircularLinks( outputNodeEntry, 'forward', inputNodeEntry );
		this.checkForCircularLinks( inputNodeEntry, 'backward', outputNodeEntry );
		this.checkForCircularLinks( inputNodeEntry, 'forward', outputNodeEntry );

		outputNodeEntry.addOutput( inputNodeEntry );
		inputNodeEntry.addInput( outputNodeEntry );
	}


	private checkForCircularLinks( entry : NodeCacheEntry, direction : string, targetEntry : NodeCacheEntry )
	{
		const linkTestFn = ( link : NodeCacheEntry ) => {
			if( link === targetEntry )
			{
				throw new Error( `Circular graph of nodes detected` );
			}
		};

		this.traverse( entry, direction, linkTestFn );
	}


	public traverse( entry : NodeCacheEntry, direction : string, callback : Function )
	{
		_.each(
			( direction === 'forward' ) ? entry.outputs : entry.inputs,
			( link : NodeCacheEntry ) => {
				callback( link );
				this.traverse( link, direction, callback );
			}
		);
	}


	public unlink( outputNode : Node, inputNode : Node )
	{
		this.canModify();

		const outputNodeEntry	= this.find( outputNode ),
			inputNodeEntry		= this.find( inputNode );

		outputNodeEntry.removeOutput( inputNodeEntry );
		inputNodeEntry.removeInput( outputNodeEntry );
	}


	public remove( node : Node )
	{
		this.canModify();

		const entry = this.find( node );

		_.each(
			this.nodes,
			( e : NodeCacheEntry ) => {
				e.removeInput( entry );
				e.removeOutput( entry );
			}
		);

		_.remove( this.nodes, ( e : NodeCacheEntry ) => ( e === entry ) );
	}


	public exists( node : Node )
	{
		try
		{
			this.find( node );

			return true;
		}
		catch( e )
		{
			if( e.match( /Unknown node/ ) )
			{
				return false;
			}

			throw e;
		}
	}


	public find( node : Node|string|number ) : NodeCacheEntry
	{
		if( _.isNumber( node ) === true )
		{
			return this.nodes[ <number>node ];
		}

		let entry;

		if( _.isString( node ) === true )
		{
			entry = _.find( this.nodes, ( entry : NodeCacheEntry ) => ( entry.node.name === <string>node ) );
		}
		else
		{
			entry = _.find( this.nodes, ( entry : NodeCacheEntry ) => ( entry.node === <Node>node ) );
		}

		if( _.isUndefined( entry ) )
		{
			throw new Error( `Unknown node: ${node}` );
		}

		return entry;
	}


	protected canModify()
	{
		if( this.compiled )
		{
			throw new Error( `Node graph cannot be modified after compilation` );
		}
	}


	public compile()
	{
		this.compiled = true;
	}

}


export default NodeCache;
