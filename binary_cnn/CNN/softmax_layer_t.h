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
	tensor_t<float> grads_in;
	tensor_t<float> in;
	tensor_t<float> out;
	bool to_normalize;
	bool debug, clip_gradients_flag;	
	softmax_layer_t( tdsize in_size, bool to_normalize=true,bool clip_gradients_flag = true, bool debug_flag = false):
		in( in_size.m, in_size.x, 1, 1),
		out( in_size.m, in_size.x, 1, 1 ),
		grads_in( in_size.m, in_size.x, 1, 1)
	{
		this->to_normalize = to_normalize;
		this->debug = debug_flag;
		this->clip_gradients_flag = clip_gradients_flag;
	}
	
	void activate( tensor_t<float>& in )
	{	
		this->in = in;
		activate();
	}
	
	void normalize_in()
	{
		for(int tm=0; tm<in.size.m; tm++){
			float minm = 1e7;
			float maxm = -1e7;
			for(int i=0; i<in.size.x; i++){
				minm = min(in(tm,i,0,0), minm);
				maxm = max(in(tm,i,0,0), maxm);
			}

			for(int i=0; i<in.size.x; i++){
				in(tm,i,0,0) = (in(tm,i,0,0) - minm)/(maxm-minm);
			// in(tm,i,0,0) = in(tm,i,0,0) - maxm;
			}
		}
	}
	
	void activate()
	{	
		if (to_normalize)
			normalize_in();
		

		float temp1, sum = 0;
		for ( int tm = 0; tm < in.size.m; tm++ ){
			sum = 0;
			for ( int i = 0; i < in.size.x; i++ )
			{
				temp1= in(tm, i, 0, 0);
				// sum += exp(temp1);
				sum += max(exp(temp1), float(1e-7));
			}
		
			for ( int i = 0; i < in.size.x; i++ )
				// out(tm, i, 0, 0) = exp(in(tm, i, 0, 0);
				out(tm, i, 0, 0) = max(exp(in(tm, i, 0, 0)), float(1e-7))/sum;		
		}
		
		if(debug)
		{		
			cout<<"********output for softmax ********\n";
			print_tensor(out);
		}
	}
	
	
	void fix_weights(float learning_rate)
	{
		
	}
	
	void calc_grads( tensor_t<float>& grad_next_layer )
	{
		float m = out.size.m;
		for(int e = 0; e<out.size.m; e++){
			int idx;
			for(int i=0; i<out.size.x; i++){
				if(int(grad_next_layer(e,i,0,0)) == 1){
					idx = i;
					grads_in(e,i,0,0) = -(1-out(e,i,0,0))/m;
				}
			}
			for(int i=0; i<out.size.x; i++){
				if(idx!=i){
					grads_in(e,i,0,0) = ((out(e,i,0,0))/m);
				}
				clip_gradients(clip_gradients_flag, grads_in(e,i,0,0));
				 
			}	
		}
		
		if(debug)
		{
			cout<<"********grads_in for softmax*********\n";
			print_tensor(grads_in);
		}
	}	
};
#pragma pack(pop)