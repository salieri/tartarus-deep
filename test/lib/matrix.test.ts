import { Matrix } from '../../lib/math';


describe( 'Matrix',
	function()
	{
		it( 'should transpose matrices',
			function()
			{
				const m = new Matrix(
					[
						[ 1, 2, 3, 4, 5 ],
						[ 9, 8, 7, 6, 5 ]
					]
				);

				m.transpose().toJSON().should.deep.equal(
					[
						[ 1, 9 ],
						[ 2, 8 ],
						[ 3, 7 ],
						[ 4, 6 ],
						[ 5, 5 ]
					]
				);
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

				( () => m1.matmul( m2 ) ).should.Throw( /Cannot multiply matrices where a.cols does not match b.rows/ );
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

				const result = m1.matmul( m2 );

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
