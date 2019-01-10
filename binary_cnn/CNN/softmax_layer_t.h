/*
To implement softmax 
*/

#pragma once
#include <math.h>
#include <float.h>
#include <string.h>
#include "layer_t.h"
#include "optimization_method.h"
#include "gradient_t.h"
#include "tensor_bin_t.h"

#pragma pack(push, 1)
struct softmax_layer_t
{
	layer_type type = layer_type::softmax ;
	
	tensor_t<float> in;
	tensor_t<float> out;
	tensor_t<float> grads_in;
	
	softmax_layer_t( tdsize in_size):
	in( in_size.x, 1, 1),
		out( in_size.x, 1, 1 ),
		grads_in( in_size.x, in_size.y, in_size.z ),
	{
	
	}
	
	void activate( tensor_t<float>& in )
	{
		this->in = in;
		activate();
	}
	
	
	void activate()
	{
	
		float temp1,temp2,sum;
		for ( int i = 0; i < in.size.x; i++ )
		{
			temp1= in(i,0,0);
			sum+=exp(temp1);
		}
		
		for ( int i = 0; i < in.size.x; i++ )
		{
			out(i,0,0) = exp(in(i,0,0))/sum;
			
		}
				
				
		cout<<"********output for softmax ********\n";
		print_tensor(out);
	}
	
	
	void fix_weights()
	{

	}
	
	void calc_grads( tensor_t<float>& grad_next_layer )
	{
				
	}
	
	
	
}