#pragma once
#include <math.h>
#include <float.h>
#include <string.h>
#include "layer_t.h"
#include "optimization_method.h"
#include "gradient_t.h"

#pragma pack(push, 1)
struct fc_layer_t
{
	layer_type type = layer_type::fc;
	tensor_t<float> in;
	tensor_t<float> weights;
	tensor_t<gradient_t> weights_grad;
	tdsize in_size, out_size;
	bool train = false;
	bool debug, clip_gradients_flag;

	fc_layer_t( tdsize in_size, tdsize out_size,  bool clip_gradients_flag=false, bool debug_flag = false )
		:
		weights( in_size.x*in_size.y*in_size.z, out_size.x, 1, 1 )
	{

		this->in_size = in_size;
		this->out_size = out_size;
		this->debug = debug_flag;
		this->clip_gradients_flag = clip_gradients_flag;
		int maxval = in_size.x * in_size.y * in_size.z;
	
		// Weight initialization
		
		for ( int i = 0; i < out_size.x; i++ )
			for ( int h = 0; h < in_size.x * in_size.y * in_size.z; h++ )
				weights( h, i, 0, 0 ) =  ( 1.0f * (rand()-rand())) / (float( RAND_MAX ) * 10);  // Generates a random number between -1 and 1 
		
		if(debug)
		{
			cout << "********weights for fc************\n";
			print_tensor(weights);
		}

	}

	int map( point_t d )
	//  Maps weight unit to corresponding input unit.
	{
		return d.m * (in_size.x * in_size.y * in_size.z) +
			d.z * (in_size.x * in_size.y) +
			d.y * (in_size.x) +
			d.x;
	}

	tensor_t<float> activate( tensor_t<float>& in, bool train )
	/* 
	 * `activate` activates (forward propogate) the fc layer.
	 * It saves the result after propogation in `out` variable.
	 */
	{
		
		if ( train ) this->in = in;  	// Only save `in` while training to save RAM during inference
		tensor_t<float> out( in.size.m, weights.size.x, 1, 1 );
		int xyz = in.size.x * in.size.y * in.size.z;
		for ( int e = 0; e < in.size.m; e++){
			int exyz = e*xyz;
			for ( int n = 0; n < weights.size.x; n++ )
			{
				float inputv = 0;
						for ( int i = 0; i < xyz; i++ )
						{
							inputv += in.data[exyz +i] * weights.data[ weights.size.x*i + n]; 
						}	
				out( e, n, 0, 0 ) = inputv;
			}
		
		}

		
		if(debug)
		{
			cout<<"*******output for fc**********\n";
			print_tensor(out);
		}
		return out;
	}

	void fix_weights(float learning_rate)
	{
		for ( int n = 0; n < weights.size.x; n++ )
			for ( int i = 0; i < in_size.x; i++ )
				for ( int j = 0; j < in_size.y; j++ )
					for ( int z = 0; z < in_size.z; z++ )
					{
						int m = map( { 0, i, j, z } );
						float& w = weights( m, n, 0, 0 );
						
						gradient_t& grad = weights_grad( m, n, 0, 0 ) ;
						w = update_weight( w, grad, 1, false, learning_rate);
						update_gradient( grad );
					}

		if(debug)
		{
			cout<<"*******new weights for float fc*****\n";
			print_tensor(weights);
		}
	}

	tensor_t<float> calc_grads( tensor_t<float>& grad_next_layer )
	
	// Calculates backward propogation and saves result in `grads_in`. 
	{
		assert(in.size > 0);
		tensor_t<float> grads_in( grad_next_layer.size.m, in_size.x, in_size.y, in_size.z );
		weights_grad.resize(weights.size);

		int xyz = in.size.x * in.size.y * in.size.z;
		for(int e=0; e<grad_next_layer.size.m; e++){
			int exyz = e*xyz;
			int en = e*weights.size.x;
			for ( int n = 0; n < weights.size.x; n++ )
			{
				int enn = en + n;
				for ( int i = 0; i < xyz; i++ )
						{
							grads_in.data[ exyz + i ] += grad_next_layer.data[enn] * weights.data[ weights.size.x*i + n ];
							weights_grad.data[ weights.size.x*i + n ].grad += grad_next_layer.data[enn] * in.data[ exyz + i ];
							// clip_gradients(clip_gradients_flag, grads_in.data[ exyz + i ]);
						}
			}
		}	
		if(debug)
		{
			cout<<"**********grads_in for float fc***********\n";
			print_tensor(grads_in);
		}
		return grads_in;	

	}
	
	void save_layer( json& model ){
		model["layers"].push_back( {
			{ "layer_type", "fc" },
			{ "in_size", {in_size.m, in_size.x, in_size.y, in_size.z} },
			{ "out_size", {out_size.m, out_size.x, out_size.y, out_size.z} },
			{ "clip_gradients", clip_gradients_flag}
		} );
	}

	void save_layer_weight( string fileName ){
		vector<float> data;
		int m = weights.size.m;
		int x = weights.size.x;
		int y = weights.size.y;
		int z = weights.size.z;
		int array_size = m * x * y * z;
		for ( int i = 0; i < array_size; i++ )
			data.push_back(weights.data[i]);

		ofstream file(fileName);
		json weight = { 
			{ "type", "fc" },
			{ "size", array_size },
			{ "data", data}
		};
		file << weight << endl;
		file.close();
	}

	void load_layer_weight(string fileName){
		ifstream file(fileName);
		json weight;
		file >> weight;
		assert(weight["type"] == "fc");
		vector<float> data = weight["data"];
		int size  = weight["size"];
		for (int i = 0; i < size; i++)
			this->weights.data[i] = data[i];
		file.close();
	}

	void print_layer(){
		cout << "\n\n FC Layer : \t";
		cout << "\n\t in_size:\t";
		print_tensor_size(in_size);
		cout << "\n\t out_size:\t";
		print_tensor_size(out_size);
	}
};
#pragma pack(pop)
