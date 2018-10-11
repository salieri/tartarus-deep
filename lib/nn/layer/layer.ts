import { NDArray } from '../../ndarray';
import Joi from 'joi';


export interface LayerParams {
	[key: string]: any;
}

export interface LayerDescriptor {
	[key: string]: any;
}


export class Layer
{
	public params	: LayerParams;
	public name		: string;

	private static layerCounter : number = 0;

	constructor( params : LayerParams = {}, name? : string )
	{
		this.params	= this.validateParams( params );
		this.name	= name ? name : `${this.constructor.name}#${Layer.layerCounter++}`;
	}


	private validateParams( params : LayerParams ) : LayerParams
	{
		const result = Joi.validate( params, this.getDescriptor() );

		if( result.error )
		{
			throw result.error;
		}

		return result.value;
	}


	public setParam( paramName : string, value : any ) : Layer
	{
		const result = Joi.validate( this.params[ paramName ], value );

		if( result.error )
		{
			throw result.error;
		}

		this.params[ paramName ] = result.value;

		return this;
	}


	public getDescriptor() : LayerDescriptor
	{
		return {};
	}


	public calculate( x : NDArray ) : NDArray
	{
		return x;
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


