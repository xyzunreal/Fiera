/* TODO: Implement prelu instead of relu activation function
   It follows:
    f(x) = alpha * x	for x < 0
    f(x) = x		for x >= 0
    where alpha is a learnable parameter

*/
#pragma once
#include "layer_t.h"



/*
 * referencing paramteric-Relu
 * args::
 * 	in(tensor)::accepts the input layer
 *
 * return:: 
 * 	out(tensor)::layer with elements as max(x,a*x) 
 * 
 * 	
*/

#pragma pack(push, 1)
struct prelu_layer_t
{
	layer_type type = layer_type::prelu;
	tensor_t<float> grads_in;
	tensor_t<float> in;
	tensor_t<float> out;
	float alpha;

	prelu_layer_t( tdsize in_size ):
		in( in_size.x, in_size.y, in_size.z ),
		out( in_size.x, in_size.y, in_size.z ),
		grads_in( in_size.x, in_size.y, in_size.z )
	{
		alpha=0.05;
	}


	void activate( tensor_t<float>& in )
	{
		this->in = in;
		activate();
	}

	void activate()
	{
		for ( int i = 0; i < in.size.x; i++ )
			for ( int j = 0; j < in.size.y; j++ )
				for ( int z = 0; z < in.size.z; z++ )
				{
					float x = in( i, j, z );
					if ( x < 0 )
						x = x*alpha;
					out( i, j, z ) = x;
				}
		cout<<"********output for prelu*****\n";
		print_tensor(out);
	}

	void fix_weights()
	{

	}

	void calc_grads( tensor_t<float>& grad_next_layer )
	{
		for ( int i = 0; i < in.size.x; i++ )
			for ( int j = 0; j < in.size.y; j++ )
				for ( int z = 0; z < in.size.z; z++ )
				{
					grads_in( i, j, z ) = (in( i, j, z ) < 0) ?
						(in( i, j, z ) * grad_next_layer( i, j, z )) : (0);
				}
							
	}
};
#pragma pack(pop)