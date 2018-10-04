import 'mocha';
import { expect } from 'chai';

import NDArray from '../../lib/ndarray';


describe( 'N-dimensional Array',
	function()
	{
		it( 'should not create NDArray without specified dimensions',
			function()
			{
				( () => ( new NDArray() ) ).should.Throw( /Unspecified dimensions/ );
			}
		);


		it( 'should create NDArray from dimension spec',
			function()
			{
				const m = new NDArray( 3, 4 );

				m.countDims().should.equal( 2 );
				m.getDims()[ 0 ].should.equal( 3 );
				m.getDims()[ 1 ].should.equal( 4 );

				m.get().length.should.equal( 3 );
				(<number[]>m.get()[ 1 ] ).length.should.equal( 4 );
			}
		);


		it( 'should create NDArray directly from data spec',
			function()
			{
				const m = new NDArray(
					[
						[ 0, 0, 0, 0 ],
						[ 0, 1, 1, 0 ],
						[ 0, 0, 0, 0 ]
					]
				);

				m.countDims().should.equal( 2 );
				m.getDims()[ 0 ].should.equal( 3 );
				m.getDims()[ 1 ].should.equal( 4 );

				m.get().length.should.equal( 3 );
				(<number[]>m.get()[ 1 ] ).length.should.equal( 4 );

				m.getAt( [ 0, 0 ] ).should.equal( 0 );
				m.getAt( [ 1, 1 ] ).should.equal( 1 );
				m.getAt( [ 1, 2 ] ).should.equal( 1 );
				m.getAt( [ 2, 3 ] ).should.equal( 0 );
			}
		);


		it( 'should create NDArray from another NDArray',
			function()
			{
				const mSource = new NDArray(
					[
						[ 0, 0, 0, 0 ],
						[ 0, 1, 1, 0 ],
						[ 0, 0, 0, 0 ]
					]
				);

				const m = new NDArray( mSource );

				m.countDims().should.equal( 2 );
				m.getDims()[ 0 ].should.equal( 3 );
				m.getDims()[ 1 ].should.equal( 4 );

				m.get().length.should.equal( 3 );
				(<number[]>m.get()[ 1 ] ).length.should.equal( 4 );

				m.getAt( [ 0, 0 ] ).should.equal( 0 );
				m.getAt( [ 1, 1 ] ).should.equal( 1 );
				m.getAt( [ 1, 2 ] ).should.equal( 1 );
				m.getAt( [ 2, 3 ] ).should.equal( 0 );
			}
		);


		it( 'should be comparable with other NDArrays',
			function()
			{
				const m1 = new NDArray(
					[
						[ 0, 0, 0, 0 ],
						[ 0, 1, 1, 0 ],
						[ 0, 0, 0, 0 ]
					]
				);

				const m2 = new NDArray(
					[
						[ 0, 0, 0, 0 ],
						[ 0, 1, 1, 0 ],
						[ 0, 0, 0, 0 ]
					]
				);

				const n1 = new NDArray(
					[
						[ 3, 0, 0, 0 ],
						[ 0, 3, 3, 0 ],
						[ 0, 0, 0, 3 ]
					]
				);

				m1.equals( m1 ).should.equal( true );
				m1.equals( m2 ).should.equal( true );
				m2.equals( m1 ).should.equal( true );

				m1.equals( n1 ).should.equal( false );
				n1.equals( n1 ).should.equal( true );
			}
		);
	}
);

