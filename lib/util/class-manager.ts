import _ from 'lodash';

export interface ClassModule {
	[key: string]: any;
}

export interface InstanceParams {
	[key: string]: any;
}


export class ClassManager
{
	protected knownClasses : ClassModule;
	protected baseClass : any;

	constructor( knownClasses : ClassModule, baseClass : any )
	{
		this.baseClass		= baseClass;
		this.knownClasses	= this.prepareClasses( knownClasses );
	}


	protected prepareClasses( knownClasses : ClassModule ) : ClassModule
	{
		const filteredClasses : ClassModule = {};

		_.each(
			knownClasses,
			( cls, key ) => {
				if( cls.prototype instanceof this.baseClass )
				{
					filteredClasses[ ClassManager.getSimplifiedName( key ) ] = cls;
				}
			}
		);

		return filteredClasses;
	}


	public coerce( instanceDefinition : string|object, params? : InstanceParams )
	{
		if( instanceDefinition instanceof this.baseClass )
		{
			return instanceDefinition;
		}

		if( _.isString( instanceDefinition ) === false )
		{
			throw new Error( 'Cannot coerce: Invalid data' );
		}

		return this.factory( <string>instanceDefinition, params );
	}


	public factory( className : string, params? : InstanceParams ) : any
	{
		const ClassSpec = this.find( className );

		return new ClassSpec( params );
	}


	public find( className : string ) : any
	{
		const simplifiedName = ClassManager.getSimplifiedName( className );

		if( this.knownClasses.hasOwnProperty( simplifiedName ) === false )
		{
			throw new Error( 'Unknown class: `className`' );
		}

		return this.knownClasses[ simplifiedName ];
	}


	public static getSimplifiedName( className : string ) : string
	{
		return className.toLowerCase().replace( /[^a-z0-9]/g, '' );
	}

}

