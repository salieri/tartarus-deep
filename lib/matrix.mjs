
import NDArray from './ndarray.mjs';
import _ from 'lodash';

class Matrix extends NDArray
{

	constructor( ...dimensions )
	{
		super( ...dimensions );
	}


	validateConstructor( dimensions )
	{
		super.validateConstructor( dimensions );


	}

}


export default Matrix;

