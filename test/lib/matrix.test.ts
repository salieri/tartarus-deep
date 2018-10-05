import Matrix from '../../lib/matrix';


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
	}
);
