import {ClassManager} from '../../../lib/util/class-manager';
import {Activation} from '../../../lib/nn/activation';
import * as Activations from '../../../lib/nn/activation';


describe( 'Class Manager',
	function()
	{
		it( 'should instantiate any class in the module',
			function()
			{
				const cm		= new ClassManager( Activations, Activation ),
					activator	= cm.factory( 'binary' );

				activator.should.be.instanceOf( Activations.Binary );
			}
		);


		it( 'should instantiate a class with parameters',
			function()
			{
				const cm		= new ClassManager( Activations, Activation ),
					activator	= cm.factory( 'ReLU', { leak : 1.23 } );

				activator.should.be.instanceOf( Activations.ReLU );
				activator.params.leak.should.equal( 1.23 );
			}
		);


		it( 'should return the instance, if one is passed to the coerce function',
			function()
			{
				const cm		= new ClassManager( Activations, Activation ),
					activator	= cm.coerce( 'bent-identity' );

				activator.should.be.instanceOf( Activations.BentIdentity );

				cm.coerce( activator ).should.equal( activator );
				cm.coerce( 'bent-identity' ).should.not.equal( activator );
			}
		);

	}

);
