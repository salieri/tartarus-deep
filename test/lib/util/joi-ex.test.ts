import {JoiEx, ClassManager} from '../../../lib/util';
import {Activation} from '../../../lib/nn/activation';
import * as activations from '../../../lib/nn/activation';

import { expect } from 'chai';


describe( 'JoiEx',
	function()
	{
		it( 'should pass instantiated objects through coercing validators',
			function()
			{
				const cm 		= new ClassManager( activations, Activation ),
					schema		= JoiEx.activation(),
					activation	= cm.factory( 'sigmoid' ),
					result		= JoiEx.validate( activation, schema );

				expect( result.error ).to.equal( null );
				expect( result.value ).to.equal( activation );
				expect( result.value ).to.be.instanceOf( Activation );
				expect( result.value ).to.be.instanceOf( activations.Sigmoid );
			}
		);

		it( 'should not pass instantiated objects through coercing validators',
			function()
			{
				const cm 		= new ClassManager( activations, Activation ),
					schema		= JoiEx.activation(),
					activation	= { moo : 'moo' },
					result		= JoiEx.validate( activation, schema );

				expect( result.error ).to.match( /Unexpected data type passed to the coerce function/ );
			}
		);


		it( 'should instantiate class names passed to coerce() validator',
			function()
			{
				const cm	= new ClassManager( activations, Activation ),
					schema	= JoiEx.activation(),
					result	= JoiEx.validate( 'sigmoid', schema );

				expect( result.error ).to.equal( null );
				expect( result.value ).to.be.instanceOf( Activation );
				expect( result.value ).to.be.instanceOf( activations.Sigmoid );
			}
		);


		it( 'should not instantiate unknown class names passed to coerce() validator',
			function()
			{
				const cm	= new ClassManager( activations, Activation ),
					schema	= JoiEx.activation(),
					result	= JoiEx.validate( 'this-does-not-exist', schema );

				expect( result.error ).to.match( /Unexpected data type passed to the coerce function/ );
			}
		);
	}
);
