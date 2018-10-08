


```js
config = {
	header : {
		name	: '',
		version	: '',
		description : ''
	},
	
	input : {
		type : '',
		params : {
		
		}
	},
	
	layers : [
		{
			name: 'layer-1',
			type: 'nn',
			params : {
				size : 800
			}
		},
		
		{
			name : 'layer-2',
			type : 'nn',
			params : {
				size : 400
			}
		},
		
		{
			name : 'relu-1',
			type : 'relu'
		}
	],
	
	output : {
		type : '',
		params : {
		
		}
	}
}


```
