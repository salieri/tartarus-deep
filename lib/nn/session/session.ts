import {create as createRandomSeed, RandomSeed} from 'random-seed';


export class Session
{
	public rand : RandomSeed;

	constructor( seed? : string )
	{
		this.rand = createRandomSeed( seed );
	}
}

