/*Implement prelu instead of relu activation function
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
	gradient_t grads_alpha;
	float p_relu_zero;

	prelu_layer_t( tdsize in_size ):
		in( in_size.m, in_size.x, in_size.y, in_size.z ),
		out(in_size.m, in_size.x, in_size.y, in_size.z ),
		grads_in( in_size.m, in_size.x, in_size.y, in_size.z )
	{
		alpha=0.05;
		p_relu_zero = 0.5;
	}


	void activate( tensor_t<float>& in )
	{
		this->in = in;
		activate();
	}

	void activate()
	{
		for ( int tm = 0; tm < in.size.m; tm++ )
			for ( int i = 0; i < in.size.x; i++ )
				for ( int j = 0; j < in.size.y; j++ )
					for ( int k = 0; k < in.size.z; k++ )
					{
						float x = in( tm, i, j, k);
						if ( x < 0 )
							x = x*alpha;
						out( tm, i, j, k) = x;
					}
		cout<<"********output for prelu*****\n";
		print_tensor(out);
	}

	void fix_weights()
	{	// grads_alpha contains sum of gradients of alphas for all examples. 
		grads_alpha.grad /= out.size.m;
		alpha = update_weight(alpha,grads_alpha);
		update_gradient(grads_alpha);

		cout<<"*******updated alpha for prelu*****\n";
		cout<<alpha<<endl;
	}

	void calc_grads( tensor_t<float>& grad_next_layer )
	{
		for ( int e = 0; e < in.size.m; e++ ){
			for ( int i = 0; i < in.size.x; i++ ){
				for ( int j = 0; j < in.size.y; j++ ){
					for ( int k = 0; k < in.size.z; k++ )
					{
						// grads_in( e, i, j, k) = (in( e, i, j, k) < 0) ? (in( tm, i, j, k) * grad_next_layer( tm, i, j, k)) : (0);
						if(in(e,i,j,k)>0){
							grads_in(e,i,j,k) = grad_next_layer(e,i,j,k);
						}
						else if(int(in(e,i,j,k)) == 0){
							grads_in(e,i,j,k) = grad_next_layer(e,i,j,k)*p_relu_zero;
						}
						else{
							grads_alpha.grad += grad_next_layer(e,i,j,k) * (in(e, i, j, k));
							grads_in(e,i,j,k) = grad_next_layer(e,i,j,k)*alpha;
						}
					}
				}
			
			}
		}

		cout<<"***********grads_in for prelu********\n";
        print_tensor(grads_in);
							
	}
};
#pragma pack(pop)