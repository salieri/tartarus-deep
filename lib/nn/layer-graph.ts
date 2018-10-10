import _ from 'lodash';
import Layer from './layer';
import LayerGraphNode from './layer-graph-node';


class LayerGraph
{
	protected nodes		: LayerGraphNode[] = [];
	protected compiled	: boolean = false;


	/**
	 * Attach layer to the last entry in the current graph.
	 * Essentially creates a sequential graph.
	 * @param layer
	 */
	public push( layer : Layer ) : LayerGraphNode
	{
		let parentLayer;

		if( this.nodes.length > 0 )
		{
			parentLayer = this.nodes[ this.nodes.length - 1 ].layer;
		}

		return this.add( layer, parentLayer );
	}


	/**
	 * Add a layer to the graph
	 * @param layer
	 * @param [parentLayer] If specified, connects `layer` to the output of `parentLayer`
	 */
	public add( layer : Layer, parentLayer? : Layer ) : LayerGraphNode
	{
		this.canModify();

		if( this.exists( layer ) === true )
		{
			throw new Error( `Layer '${layer.name}' already exists in this graph` );
		}

		if( parentLayer )
		{
			if( this.exists( parentLayer ) === false )
			{
				throw new Error( `Parent layer '${layer.name}' does not exist in this graph` );
			}

			if( parentLayer === layer )
			{
				throw new Error( `Parent layer cannot be same instance than the layer being added` );
			}
		}

		const node = new LayerGraphNode( layer );

		this.nodes.push( node );

		if( parentLayer )
		{
			this.link( parentLayer, layer );
		}

		return node;
	}


	/**
	 * Link two neurons
	 * @param outputLayer
	 * @param inputLayer
	 */
	public link( outputLayer : Layer, inputLayer : Layer ) : void
	{
		this.canModify();

		const outputNode	= this.find( outputLayer ),
			inputNode		= this.find( inputLayer );

		this.checkForCircularLinks( outputNode, 'backward', inputNode );
		this.checkForCircularLinks( outputNode, 'forward', inputNode );
		this.checkForCircularLinks( inputNode, 'backward', outputNode );
		this.checkForCircularLinks( inputNode, 'forward', outputNode );

		outputNode.addOutput( inputNode );
		inputNode.addInput( outputNode );
	}


	/**
	 * Test the graph does not become circular
	 * @param node
	 * @param direction
	 * @param targetNode
	 */
	private checkForCircularLinks( node : LayerGraphNode, direction : string, targetNode : LayerGraphNode ) : void
	{
		const linkTestFn = ( linkNode : LayerGraphNode ) => {
			if( linkNode === targetNode )
			{
				throw new Error( `Circular graph of nodes detected` );
			}
		};

		this.traverse( node, direction, linkTestFn );
	}


	/**
	 * Traverse the graph starting from `node`
	 * @param node
	 * @param direction Direction to traverse (`'backward'` or `'forward'`)
	 * @param {Function(LayerGraphNode node)} callback Callback to be called for every node traversed
	 */
	public traverse( node : LayerGraphNode, direction : string, callback : Function ) : void
	{
		_.each(
			( direction === 'forward' ) ? node.outputs : node.inputs,
			( linkNode : LayerGraphNode ) => {
				callback( linkNode );
				this.traverse( linkNode, direction, callback );
			}
		);
	}


	/**
	 * Unlink two layers
	 * @param outputLayer
	 * @param inputLayer
	 */
	public unlink( outputLayer : Layer, inputLayer : Layer ) : void
	{
		this.canModify();

		const outputNode	= this.find( outputLayer ),
			inputNode		= this.find( inputLayer );

		outputNode.removeOutput( inputNode );
		inputNode.removeInput( outputNode );
	}


	/**
	 * Remove layer from the graph
	 * @param layer
	 */
	public remove( layer : Layer ) : void
	{
		this.canModify();

		const node = this.find( layer );

		_.each(
			this.nodes,
			( e : LayerGraphNode ) => {
				e.removeInput( node );
				e.removeOutput( node );
			}
		);

		_.remove( this.nodes, (e : LayerGraphNode ) => ( e === node ) );
	}


	/**
	 * Check if a layer exists in the graph
	 * @param layer
	 */
	public exists( layer : Layer ) : boolean
	{
		try
		{
			this.find( layer );

			return true;
		}
		catch( e )
		{
			if( e.match( /Unknown layer/ ) )
			{
				return false;
			}

			throw e;
		}
	}


	/**
	 * Find layer in the graph
	 * @param {Layer|string|number} layer If `string`, matches against the name of the layer;
	 * if `number` accesses graph layer pool by index;
	 * if `Layer` finds the node that matches the specific instance
	 */
	public find( layer : Layer|string|number ) : LayerGraphNode
	{
		if( _.isNumber( layer ) === true )
		{
			return this.nodes[ <number>layer ];
		}

		let node;

		if( _.isString( layer ) === true )
		{
			node = _.find( this.nodes, (n : LayerGraphNode ) => ( n.layer.name === <string>layer ) );
		}
		else
		{
			node = _.find( this.nodes, (n : LayerGraphNode ) => ( n.layer === <Layer>layer ) );
		}

		if( _.isUndefined( node ) )
		{
			throw new Error( `Unknown layer: ${layer}` );
		}

		return node;
	}


	protected canModify() : void
	{
		if( this.compiled )
		{
			throw new Error( `Node graph cannot be modified after compilation` );
		}
	}


	public compile() : void
	{
		this.canModify();

		this.compiled = true;
	}

}


export default LayerGraph;
