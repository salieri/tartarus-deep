import { Initializer, InitializerDescriptor, InitializerParams } from './initializer';
import { NDArray } from '../../math';
import Joi from 'joi';
import {create as createRandomSeed, RandomSeed} from 'random-seed';


export class RandomUniform extends Initializer {

	private rand : RandomSeed;

	constructor( params : InitializerParams )
	{
		super( params );

		this.rand = createRandomSeed( this.params.seed );
	}


	public initialize( data : NDArray ) : NDArray
	{
		return data.apply( () : number => ( this.rand.floatBetween( this.params.min, this.params.max ) ) );
	}


	public getDescriptor() : InitializerDescriptor
	{
		return {
			min		: Joi.number().default( 0.0 ).description( 'Minimum random value' ),
			max		: Joi.number().default( 1.0 ).description( 'Maximum random value' ),
			seed	: Joi.string().optional().description( 'Random seed' )
		};
	}
}
