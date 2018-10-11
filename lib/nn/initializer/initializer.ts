import { NDArray } from '../../math';

export interface InitializerParams {
	[key: string]: any
}

export interface InitializerDescriptor {
	[key: string]: any
}


export class Initializer
{
	protected params : InitializerParams;

	constructor( params : InitializerParams = {} )
	{
		this.params = params;
	}


	public initialize( data : NDArray ) : NDArray
	{
		throw new Error( 'Not implemented' );
	}


	public getDescriptor() : InitializerDescriptor
	{
		return {};
	}

}

