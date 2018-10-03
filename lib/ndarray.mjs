import _ from 'lodash';

/**
 * new Matrix( 20 )
 * new Matrix( 20, 30, 40, 50 )
 */

class NDArray
{
	/**
	 * @param {Integer...|NDArray|Number[]} dimensions
	 */
	constructor( ...dimensions )
	{
		this.validateConstructor( dimensions );

		if( dimensions.length === 1 )
		{
			const dimEl = dimensions[ 0 ];

			if( dimEl instanceof NDArray )
			{
				dimEl.clone( this );
				return;
			}
			else if( _.isArray( dimEl ) === true )
			{
				this.dimensions = NDArray.resolveDimensions( dimEl );
				this.data		= NDArray.createDimArray( this.dimensions );

				this.setData( dimEl );
				return;
			}
		}

		this.dimensions	= dimensions;
		this.data		= NDArray.createDimArray( this.dimensions );
		this.id			= NDArray.idCounter++;
	}


	validateConstructor( dimensions )
	{
		if( ( !dimensions ) || ( dimensions.length === 0 ) )
		{
			throw new Error( `Unspecified dimensions` );
		}
	}


	/**
	 * @param {int[]} dimensions
	 * @param {Number} [initialValue=0]
	 * @returns {Number[]} Multi-dimensional array representing the matrix
	 */
	static createDimArray( dimensions, initialValue = 0 )
	{
		function iterateCreation( depthIndex )
		{
			const size = dimensions[ depthIndex ];

			if( depthIndex === dimensions.length - 1 )
			{
				return _.fill( Array( size ), initialValue );
			}

			return _.map( Array( size ), () => iterateCreation( depthIndex + 1, initialValue, dimensions ) );
		}

		return iterateCreation( 0 );
	}


	static resolveDimensions( data )
	{
		const dimensions = [];

		let dp = data;

		while( _.isArray( dp ) === true )
		{
			dimensions.push( dp.length );

			dp = dp[ 0 ];
		}

		return dimensions;
	}


	/**
	 * Traverse all elements in the matrix
	 *
	 * @param {function(value, position)} callback Called for each element in the matrix. If the function returns a value other than
	 * undefined, the matrix element will be set to that value.
	 */
	traverse( callback )
	{
		const dimensions = this.dimensions;

		function iterateMatrix( depthIndex, positionPath, dataSource )
		{
			positionPath.push( 0 );

			const posIdx = positionPath.length - 1;

			_.each(
				dataSource,
				( dataVal, dataIdx ) => {
					positionPath[ posIdx ] = dataIdx;

					if( depthIndex === dimensions.length - 1 )
					{
						const result = callback( dataVal, positionPath );

						if( _.isUndefined( result ) === false )
						{
							dataSource[ dataIdx ] = result;
						}
					}
					else
					{
						iterateMatrix( depthIndex + 1, positionPath, dataVal );
					}
				}
			);

			positionPath.pop();
		}

		iterateMatrix( 0, [], this.data );
	}


	/**
	 * Clone a matrix
	 * @param {NDArray} [targetObj=null]
	 * @returns {NDArray}
	 * @public
	 */
	clone( targetObj )
	{
		targetObj = targetObj || new NDArray( ...this.dimensions );

		_.each(
			this,
			( v, k ) => ( targetObj[ k ] = _.cloneDeep( v ) )
		);

		targetObj.id = NDArray.idCounter++;

		return targetObj;
	}


	/**
	 * Set all matrix values to zero
	 * @public
	 */
	zero()
	{
		this.set( 0 );
	}


	/**
	 * Set all matrix values to the specified value
	 * @param {Number} value
	 * @public
	 */
	set( value )
	{
		this.traverse( () => ( value ) );
	}


	/**
	 * Get matrix data
	 * @returns {Number[]}
	 * @public
	 */
	get()
	{
		return _.cloneDeep( this.data );
	}


	/**
	 * Get value from specific position
	 * @param int[] positionPath
	 * @param {Number} value
	 */
	getAt( positionPath )
	{
		this.validatePosition( positionPath );

		return _.get( this.data, _.join( positionPath, '.' ) );
	}


	/**
	 * Set value at specific position
	 * @param int[] positionPath
	 * @param {Number} value
	 */
	setAt( positionPath, value )
	{
		this.validatePosition( positionPath );

		return _.set( this.data, _.join( positionPath, '.' ), value );
	}


	/**
	 * Set matrix data
	 * @param {Number[]} data
	 * @public
	 */
	setData( data )
	{
		this.traverse(
			( v, positionPath ) => {
				const newVal = _.get( data, _.join( positionPath, '.' ), '### not found ###' );

				if( newVal === '### not found ###' )
				{
					throw new Error( 'Invalid data shape' );
				}

				return newVal;
			}
		);
	}


	/**
	 * Validate position path
	 * @param {int[]} positionPath
	 * @private
	 */
	validatePosition( positionPath )
	{
		if( positionPath.length !== this.dimensions.length )
		{
			throw new Error( `Invalid position path: expected ${this.dimensions.length} dimensions` );
		}

		_.each(
			positionPath,
			( posIdx, dimIdx ) => {
				if( ( posIdx < 0 ) || ( posIdx > this.dimensions[ dimIdx ] ) )
				{
					throw new Error(
						`Invalid position path: Dimension ${dimIdx} position should be 0-${this.dimensions[ dimIdx ]}, was ${posIdx}`
					);
				}
			}
		);
	}


	/**
	 * Returns number of dimensions in NDArray
	 * @returns {int}
	 * @public
	 */
	countDims()
	{
		return this.dimensions.length;
	}


	/**
	 * Returns dimension sizes in NDArray
	 * @returns {int[]}
	 * @public
	 */
	getDims()
	{
		return _.cloneDeep( this.dimensions );
	}


	/**
	 * Compare two NDArrays
	 * @param {NDArray} b
	 * @returns {boolean}
	 * @public
	 */
	equals( b )
	{
		if( b.countDims() !== this.countDims() )
		{
			return false;
		}

		return _.isEqual( this.data, b.data );
	}


	toString()
	{
		return `NDArray#${this.id}: ${this.data.toString()}`;
	}


	toJSON()
	{
		return _.cloneDeep( this.data );
	}
}


NDArray.idCounter = 0;


export default NDArray;

