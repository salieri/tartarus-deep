import _ from 'lodash';

/**
 * new Matrix( 20 )
 * new Matrix( 20, 30, 40, 50 )
 */

type NumberTreeElement = number[] | number;


class NDArray
{
	private static idCounter: number = 0;

	protected id: number = 0;
	protected dimensions: number[] = [];
	protected data: NumberTreeElement[] = [];


	/**
	 * new NDArray( 20, 30, 40 ) // create n-dimensional array the size of 20x30x40
	 * new NDArray( anotherNdArray ) // create a clone of `anotherNdArray`
	 * new NDArray( [ // create a n-dimensional array from the specified array
	 * 		[ 0, 0, 0 ],
	 * 	    [ 1, 1, 1 ],
	 * 	    [ 2, 3, 4 ]
	 * 	]
	 * )
	 */
	constructor( ...dimensions: any[] )
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


	protected validateConstructor( dimensions: any[] )
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
	public static createDimArray( dimensions : number[], initialValue: number = 0 ) : NumberTreeElement[]
	{
		function iterateCreation( depthIndex: number ) : NumberTreeElement[]
		{
			const size = dimensions[ depthIndex ];

			if( depthIndex === dimensions.length - 1 )
			{
				return _.fill( Array( size ), initialValue );
			}

			return <NumberTreeElement[]> _.map( Array( size ), () => iterateCreation( depthIndex + 1 ) );
		}

		return iterateCreation( 0 );
	}


	protected static resolveDimensions( data: NumberTreeElement )
	{
		const dimensions = [];

		let dp = data;

		while( _.isArray( dp ) === true )
		{
			dp = <number[]> dp;

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
	public traverse( callback: Function )
	{
		const dimensions = this.dimensions;

		function iterateMatrix( depthIndex: number, positionPath: number[], dataSource: NumberTreeElement[] )
		{
			positionPath.push( 0 );

			const posIdx	= positionPath.length - 1,
				dsArray		= <number[]>dataSource;

			_.each(
				dsArray,
				( dataVal : NumberTreeElement, dataIdx : number ) => {
					positionPath[ posIdx ] = dataIdx;

					if( depthIndex === dimensions.length - 1 )
					{
						const result = callback( dataVal, positionPath );

						if( _.isUndefined( result ) === false )
						{
							dsArray[ dataIdx ] = result;
						}
					}
					else
					{
						iterateMatrix( depthIndex + 1, positionPath, <number[]>dataVal );
					}
				}
			);

			positionPath.pop();
		}

		iterateMatrix( 0, [], this.data );
	}


	/**
	 * Clone a matrix
	 */
	public clone( targetObj?: NDArray )
	{
		targetObj = targetObj || new NDArray( ...this.dimensions );

		_.each(
			this,
			( v, k ) => ( (<any>targetObj)[ k ] = _.cloneDeep( v ) )
		);

		targetObj.id = NDArray.idCounter++;

		return targetObj;
	}


	/**
	 * Set all matrix values to zero
	 */
	public zero()
	{
		this.set( 0 );
	}


	/**
	 * Set all matrix values to the specified value
	 */
	public set( value: number )
	{
		this.traverse( () => ( value ) );
	}


	/**
	 * Get matrix data
	 */
	public get() : number[]
	{
		return _.cloneDeep( <number[]>this.data );
	}


	/**
	 * Get value from specific position
	 */
	public getAt( positionPath: number[] )
	{
		this.validatePosition( positionPath );

		return _.get( this.data, _.join( positionPath, '.' ) );
	}


	/**
	 * Set value at specific position
	 */
	public setAt( positionPath: number[], value: number )
	{
		this.validatePosition( positionPath );

		return _.set( this.data, _.join( positionPath, '.' ), value );
	}


	/**
	 * Set matrix data
	 */
	public setData( data: NumberTreeElement )
	{
		this.traverse(
			( v : number, positionPath : number[] ) => {
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
	 */
	protected validatePosition( positionPath: number[] )
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
	 */
	public countDims() : number
	{
		return this.dimensions.length;
	}


	/**
	 * Returns dimension sizes in NDArray
	 */
	public getDims() : number[]
	{
		return _.cloneDeep( this.dimensions );
	}


	/**
	 * Compare with another NDArray
	 */
	public equals( b : NDArray )
	{
		if( b.countDims() !== this.countDims() )
		{
			return false;
		}

		return _.isEqual( this.data, b.data );
	}


	public toString() : string
	{
		return `NDArray#${this.id}: ${this.data.toString()}`;
	}


	public toJSON() : NumberTreeElement[]
	{
		return _.cloneDeep( this.data );
	}
}


export default NDArray;

