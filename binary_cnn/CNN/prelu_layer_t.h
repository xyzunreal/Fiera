/*Implement parametric ReLU activation function
   It follows:
    f(x) = alpha * x	if x < 0
    f(x) = x			if x >= 0
    where alpha is a learnable parameter

*/
#pragma once
#include "layer_t.h"

struct prelu_layer_t
{
	layer_type type = layer_type::prelu;
	tensor_t<double> grads_in;
	tensor_t<double> in;
	tensor_t<double> out;
	double alpha;
	gradient_t grads_alpha;
	double p_relu_zero;		// Differential of PReLU is undefined at 0. 'p_relu_zero' defines value to be used instead.
	bool debug,clip_gradients_flag;

	prelu_layer_t( tdsize in_size, bool clip_gradients_flag = true, bool flag_debug = false ):
	/**
	* 
	* Parameters
	* ----------
	* in_size : (int m, int x, int y, int z)
	* 		Size of input matrix.
	*
	* clip_gradients_flag : bool
	* 		Whether gradients have to be clipped or not
	* 
	* debug_flag : bool
	* 		Whether to print variables for debugging purpose
	*
	**/
	
		in( in_size.m, in_size.x, in_size.y, in_size.z ),
		out(in_size.m, in_size.x, in_size.y, in_size.z ),
		grads_in( in_size.m, in_size.x, in_size.y, in_size.z )
	{
		alpha=0.05;
		p_relu_zero = 0.5;
		this->debug=flag_debug;
		this->clip_gradients_flag = clip_gradients_flag;
	}



	void activate( tensor_t<double>& in )
	{
		this->in = in;
		activate();
	}

	void activate()

	//  `activate` FORWARD PROPOGATES AND SAVES THE RESULT IN `out` VARIABLE.
	{
		for ( int tm = 0; tm < in.size.m; tm++ )
			for ( int i = 0; i < in.size.x; i++ )
				for ( int j = 0; j < in.size.y; j++ )
					for ( int k = 0; k < in.size.z; k++ )
					{
						double x = in( tm, i, j, k);
						if ( x < 0 )
							x = x*alpha;
						out( tm, i, j, k) = x;
					}
		if(debug)
		{
			cout<<"********output for prelu*****\n";
			print_tensor(out);
		}
	}

	void fix_weights(double learning_rate)
	{
		// grads_alpha contains sum of gradients of alphas for all examples. 
		grads_alpha.grad /= out.size.m;
		alpha = update_weight(alpha,grads_alpha,1,false, learning_rate);
		update_gradient(grads_alpha);
		
		if(debug)
		{
			cout<<"*******updated alpha for prelu*****\n";
			cout<<alpha<<endl;
		}
	}

	void calc_grads( tensor_t<double>& grad_next_layer )
	{
		for ( int e = 0; e < in.size.m; e++ ){
			for ( int i = 0; i < in.size.x; i++ ){
				for ( int j = 0; j < in.size.y; j++ ){
					for ( int k = 0; k < in.size.z; k++ )
					{
						if(in(e,i,j,k)>0.0){
							grads_in(e,i,j,k) = grad_next_layer(e,i,j,k);
						}
						else if(areEqual(in(e,i,j,k),0.0)){
							grads_in(e,i,j,k) = grad_next_layer(e,i,j,k)*p_relu_zero;
					}
						else{
							grads_alpha.grad += grad_next_layer(e,i,j,k) * (in(e, i, j, k));
							grads_in(e,i,j,k) = grad_next_layer(e,i,j,k)*alpha;
						}
						clip_gradients(clip_gradients_flag, grads_alpha.grad);
						clip_gradients(clip_gradients_flag, grads_in(e,i,j,k));
					}
				}
			}
		}

		if(debug)
		{
			cout<<"***********grads_in for prelu********\n";
  	    	print_tensor(grads_in);
			cout<<"*********grad alpha***********\n";
			cout<<grads_alpha.grad<<endl;
		}
	}
};