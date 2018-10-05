import Math from '../../lib/math';
import Matrix from '../../lib/matrix';


describe( 'Math',
	function()
	{
		it( 'can multiply matrices',
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

