import _ from 'lodash';
import { Layer } from '../layer';


export class LayerGraphFeed {
	public node		: LayerGraphNode;
	public label	: string;

	constructor( node : LayerGraphNode, label : string )
	{
		this.node	= node;
		this.label	= label;
	}
}


export class LayerGraphNode
{
	layer		: Layer;
	inputs		: LayerGraphFeed[] = [];
	outputs		: LayerGraphFeed[] = [];
	connected	: boolean = false;
	level		: number = 0;


	constructor( layer : Layer )
	{
		this.layer = layer;
	}


	addOutput( node : LayerGraphNode )
	{
		this.outputs.push( node );
	}


	addInput( node : LayerGraphNode )
	{
		this.inputs.push( node );
	}


	removeOutput( node : LayerGraphNode )
	{
		_.remove( this.inputs, ( input : LayerGraphNode ) => ( node === input ) );
	}


	removeInput( node : LayerGraphNode )
	{
		_.remove( this.outputs, ( output : LayerGraphNode ) => ( node === output ) );
	}
}


