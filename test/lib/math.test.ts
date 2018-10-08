import Math from '../../lib/math';
import Matrix from '../../lib/matrix';
import NDArray from '../../lib/ndarray';


describe( 'Math',
	function()
	{
		it( 'should do elementwise add operations between n-dimensional arrays and numbers',
			function()
			{
				const nd = new NDArray(
					[
						[ 1, 2, 3 ],
						[ 4, 5, 6 ],
						[ 7, 8, 9 ]
					]
				);

				Math.elementwise.add( nd, 1 ).toJSON().should.deep.equal(
					[
						[ 2, 3, 4 ],
						[ 5, 6, 7 ],
						[ 8, 9, 10 ]
					]
				);
			}
		);


		it( 'should do elementwise subtract operations between n-dimensional arrays and numbers',
			function()
			{
				const nd = new NDArray(
					[
						[ 1, 2, 3 ],
						[ 4, 5, 6 ],
						[ 7, 8, 9 ]
					]
				);

				Math.elementwise.sub( nd, 1 ).toJSON().should.deep.equal(
					[
						[ 0, 1, 2 ],
						[ 3, 4, 5 ],
						[ 6, 7, 8 ]
					]
				);
			}
		);


		it( 'should do elementwise mul operations between n-dimensional arrays and numbers',
			function()
			{
				const nd = new NDArray(
					[
						[ 1, 2, 3 ],
						[ 4, 5, 6 ],
						[ 7, 8, 9 ]
					]
				);

				Math.elementwise.mul( nd, 2 ).toJSON().should.deep.equal(
					[
						[ 1*2, 2*2, 3*2 ],
						[ 4*2, 5*2, 6*2 ],
						[ 7*2, 8*2, 9*2 ]
					]
				);
			}
		);


		it( 'should do elementwise div operations between n-dimensional arrays and numbers',
			function()
			{
				const nd = new NDArray(
					[
						[ 2, 4, 6 ],
						[ 8, 10, 12 ],
						[ 14, 16, 18 ]
					]
				);

				Math.elementwise.div( nd, 2 ).toJSON().should.deep.equal(
					[
						[ 2/2, 4/2, 6/2 ],
						[ 8/2, 10/2, 12/2 ],
						[ 14/2, 16/2, 18/2 ]
					]
				);
			}
		);


		it( 'should do elementwise add operations between two n-dimensional arrays',
			function()
			{
				const nd = new NDArray(
					[
						[ 1, 2, 3 ],
						[ 4, 5, 6 ],
						[ 7, 8, 9 ]
					]
				);

				const nd2 = new NDArray(
					[
						[ 2, 3, 4 ],
						[ 5, 6, 7 ],
						[ 8, 9, 10 ]
					]
				);


				Math.elementwise.add( nd, nd2 ).toJSON().should.deep.equal(
					[
						[ 1+2, 2+3, 3+4 ],
						[ 4+5, 5+6, 6+7 ],
						[ 7+8, 8+9, 9+10 ]
					]
				);
			}
		);


		it( 'should do elementwise subtract operations between two n-dimensional arrays',
			function()
			{
				const nd = new NDArray(
					[
						[ 1, 2, 3 ],
						[ 4, 5, 6 ],
						[ 7, 8, 9 ]
					]
				);

				const nd2 = new NDArray(
					[
						[ 2, 3, 4 ],
						[ 5, 6, 7 ],
						[ 8, 9, 10 ]
					]
				);


				Math.elementwise.sub( nd, nd2 ).toJSON().should.deep.equal(
					[
						[ 1-2, 2-3, 3-4 ],
						[ 4-5, 5-6, 6-7 ],
						[ 7-8, 8-9, 9-10 ]
					]
				);
			}
		);


		it( 'should do elementwise mul operations between two n-dimensional arrays',
			function()
			{
				const nd = new NDArray(
					[
						[ 1, 2, 3 ],
						[ 4, 5, 6 ],
						[ 7, 8, 9 ]
					]
				);

				const nd2 = new NDArray(
					[
						[ 2, 3, 4 ],
						[ 5, 6, 7 ],
						[ 8, 9, 10 ]
					]
				);


				Math.elementwise.mul( nd, nd2 ).toJSON().should.deep.equal(
					[
						[ 1*2, 2*3, 3*4 ],
						[ 4*5, 5*6, 6*7 ],
						[ 7*8, 8*9, 9*10 ]
					]
				);
			}
		);


		it( 'should do elementwise div operations between two n-dimensional arrays',
			function()
			{
				const nd = new NDArray(
					[
						[ 1, 2, 3 ],
						[ 4, 5, 6 ],
						[ 7, 8, 9 ]
					]
				);

				const nd2 = new NDArray(
					[
						[ 2, 3, 4 ],
						[ 5, 6, 7 ],
						[ 8, 9, 10 ]
					]
				);


				Math.elementwise.div( nd, nd2 ).toJSON().should.deep.equal(
					[
						[ 1/2, 2/3, 3/4 ],
						[ 4/5, 5/6, 6/7 ],
						[ 7/8, 8/9, 9/10 ]
					]
				);
			}
		);


		it( 'should not do elementwise operations between two n-dimensional arrays of different shape',
			function()
			{
				const nd1 = new NDArray(
					[
						[ 3, 3, 3 ],
						[ 3, 3, 3 ],
						[ 3, 3, 3 ]
					]
				);

				const nd2 = new NDArray(
				[
						[ 1, 2 ],
						[ 1, 2 ]
					]
				);

				( () => Math.elementwise.add( nd1, nd2 ) ).should.Throw( /Cannot do elementwise addition on NDArrays with differing dimensions/ );
			}
		);


		it( 'should not multiply matrices of incompatible size',
			function()
			{
				const m1 = new Matrix(
					[
						[ 0, 1, 2, 3 ],
						[ 4, 5, 6, 7 ]
					]
				);

				const m2 = new Matrix(
					[
						[ 1, 2, 3 ],
						[ 4, 5, 6 ],
						[ 7, 8, 9 ]
					]
				);

				( () => Math.matmul( m1, m2 ) ).should.Throw( /Cannot multiply matrices where a.cols does not match b.rows/ );
			}
		);


		it( 'should multiply matrices',
			function()
			{
				const m1 = new Matrix(
					[
						[ 0, 1, 2, 3 ],
						[ 4, 5, 6, 7 ]
					]
				);

				const m2 = new Matrix(
					[
						[ 1, 2, 3 ],
						[ 4, 5, 6 ],
						[ 7, 8, 9 ],
						[ 10, 11, 12 ]
					]
				);

				const result = Math.matmul( m1, m2 );

				result.getRows().should.equal( 2 );
				result.getCols().should.equal( 3 );

				result.toJSON().should.deep.equal(
					[
						[ 48, 54, 60 ],
						[ 136, 158, 180 ]
					]
				);
			}
		);


	}
);

