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
	layer_type type = layer_type::softmax;
	
	tensor_t<float> in;
	tensor_t<float> out;
	tensor_t<float> grads_in;
	
	softmax_layer_t( tdsize in_size):
		in( in_size.m, in_size.x, 1, 1),
		out( in_size.m, in_size.x, 1, 1 ),
		grads_in( in_size.m, in_size.x, 1, 1)
	{

	}
	
	void activate( tensor_t<float>& in )
	{	
		cout<<"flag19\n";
		print_tensor(in);
		// cout<<in.size.m<<" "<<in.size.x<<' '<<in.size.y<<" "<<in.size.z<<endl;
		this->in = in;
		activate();
	}
	
	
	void activate()
	{

		float temp1, sum = 0;
		for ( int tm = 0; tm < in.size.m; tm++ )
			for ( int i = 0; i < in.size.x; i++ )
			{
				temp1= in(tm, i, 0, 0);
				sum+=exp(temp1);
			}
		for ( int tm = 0; tm < in.size.m; tm++ )
			for ( int i = 0; i < in.size.x; i++ )
				out(tm, i, 0, 0) = exp(in(tm, i, 0, 0))/sum;		
	
				
				
		cout<<"********output for softmax ********\n";
		print_tensor(out);
	}
	
	
	void fix_weights(float learning_rate)
	{
		
	}
	
	void calc_grads( tensor_t<float>& grad_next_layer )
	{
		for(int e = 0; e<out.size.m; e++){
			for(int i=0; i<out.size.x; i++){
				grads_in(e,i,0,0) = grad_next_layer(e,i,0,0)*(out(e,i,0,0)*(1-out(e,i,0,0)));
			}
		}

		cout<<"********grads_in for softmax*********\n";
		print_tensor(grads_in);
	}	
};
#pragma pack(pop)