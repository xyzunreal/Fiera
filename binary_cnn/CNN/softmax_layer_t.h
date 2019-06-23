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
	tdsize in_size, out_size;
	tensor_t<float> out;
	bool to_normalize;
	bool debug, clip_gradients_flag;	
	softmax_layer_t( tdsize in_size, bool to_normalize=true,bool clip_gradients_flag = true, bool debug_flag = false)
		// in( in_size.m, in_size.x, 1, 1),
		// out( in_size.m, in_size.x, 1, 1 ),
		// grads_in( in_size.m, in_size.x, 1, 1)
	{
		this->in_size = in_size;
		this->out_size = in_size;
		this->to_normalize = to_normalize;
		this->debug = debug_flag;
		this->clip_gradients_flag = clip_gradients_flag;
	}
	
	// void normalize_in()
	// {
	// 	for(int tm=0; tm<in_size.m; tm++){
	// 		float minm = 1e7;
	// 		float maxm = -1e7;
	// 		for(int i=0; i<in.size.x; i++){
	// 			minm = min(in(tm,i,0,0), minm);
	// 			maxm = max(in(tm,i,0,0), maxm);
	// 		}

	// 		for(int i=0; i<in.size.x; i++){
	// 			in(tm,i,0,0) = (in(tm,i,0,0) - minm)/(maxm-minm);
	// 		// in(tm,i,0,0) = in(tm,i,0,0) - maxm;
	// 		}
	// 	}
	// }
	
	tensor_t<float> activate( tensor_t<float> in, bool train = true)
	{	
		
		if (to_normalize) {}
			// normalize_in();
		tensor_t<float> out( in.size.m, in.size.x, 1, 1 );
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

		// if(debug)
		// {		
		// 	cout<<"********output for softmax ********\n";
		// 	print_tensor(out);
		// }

		if (train) this->out = out;
		
		return out;
	}
	
	
	void fix_weights(float learning_rate)
	{
		
	}
	
	tensor_t<float> calc_grads( tensor_t<float>& grad_next_layer )
	{
		
		float m = grad_next_layer.size.m;
		
		tensor_t<float> grads_in( m, in_size.x, 1, 1 );

		for(int e = 0; e < m; e++){
			int idx;
			for(int i=0; i<out_size.x; i++){
				if(int(grad_next_layer(e,i,0,0)) == 1){
					idx = i;
					grads_in(e,i,0,0) = -(1-out(e,i,0,0))/m;
				}
			}
			for(int i=0; i<out_size.x; i++){
		 		if(idx!=i){
					grads_in(e,i,0,0) = ((out(e,i,0,0))/m);
				}
				clip_gradients(clip_gradients_flag, grads_in(e,i,0,0));
				
			}	
		}
		
		// if(debug)
		// {
		// 	cout<<"********grads_in for softmax*********\n";
		// 	print_tensor(grads_in);
		// }

		return grads_in;
	}

	void save_layer( json& model ){
		model["layers"].push_back( {
			{ "layer_type", "softmax" },
			{ "in_size", {in_size.m, in_size.x, in_size.y, in_size.z} },
			{ "to_normalize", to_normalize },
			{ "clip_gradients", clip_gradients_flag}
		} );
	}	

	void save_layer_weight( string fileName ){
		ofstream file(fileName);
		json j = {{"type", "softmax"}};
		file << j;
		file.close();
	}

	void load_layer_weight(string fileName){

	}
	void print_layer(){
		cout << "\n\n Softmax Layer : \t";
		cout << "\n\t in_size:\t";
		print_tensor_size(in_size);
		cout << "\n\t out_size:\t";
		print_tensor_size(out_size);
	}
};
#pragma pack(pop)