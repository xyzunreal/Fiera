#pragma once
#include <math.h>
#include <float.h>
#include <string.h>
#include "layer_t.h"
#include "optimization_method.h"
#include "gradient_t.h"
#include "tensor_bin_t.h"

#pragma pack(push, 1)
struct fc_layer_bin_t
{
	layer_type type = layer_type::fc_bin;
	tensor_t<float> grads_in; 
	tensor_t<float> in;
	tensor_t<float> out;
    tensor_bin_t in_bin;
    tensor_bin_t out_bin;
    float alpha;

	std::vector<float> input;
	tensor_t<float> weights;
    tensor_bin_t weights_bin;
	tensor_t<gradient_t> gradients;

	fc_layer_bin_t( tdsize in_size, int out_size )
	/**
	 * Parameters
	 * ----------
	 * in_size : (int m, int x, int y, int z)
	 * 		Size of input matrix.
	 * 
	 * out_size : int
	 * 		No of fully connected nodes in output.
	 **/
		:
		in( in_size.m, in_size.x, in_size.y, in_size.z ),
        in_bin( in_size.m, in_size.x, in_size.y, in_size.z),
		out( in_size.m, out_size, 1, 1 ),
        out_bin( in_size.m, out_size, 1, 1),
        weights_bin( in_size.m * in_size.x * in_size.y * in_size.z, out_size, 1, 1 ),
		grads_in( in_size.m, in_size.x, in_size.y, in_size.z ),
		weights( in_size.m * in_size.x * in_size.y * in_size.z, out_size, 1, 1 ),
		gradients(in_size.m, out_size, 1, 1)
	{

		int maxval = in_size.x * in_size.y * in_size.z;

		// WEIGHT INITIALIZATION
		
		for ( int i = 0; i < out_size; i++ )
			for ( int h = 0; h < in_size.x*in_size.y*in_size.z; h++ ){
				
				weights(h,i, 0, 0 ) =  (1.0f * (rand()-rand())) / float( RAND_MAX );
				// 2.19722f = f^-1(0.9) => x where [1 / (1 + exp(-x) ) = 0.9]
				
				weights_bin.data[weights_bin(h, i, 0, 0)] = 0;
			}
	
		cout<<"***********weights for fc bin ************";
		print_tensor(weights);
	}

	void activate( tensor_t<float>& in )
	{
		this->in = in;
		activate();
	}

    void binarize()
    {
        // BINARIZING `weights`
        for (int i = 0; i < weights.size.x; ++i){
            for (int j = 0; j < weights.size.y; j++){
                weights_bin.data[weights_bin(i, j, 0, 0)] = weights(i, j, 0, 0) >= 0 ? 1 : 0;
				
			}
		}
        // BINARIZING `in`
		for ( int m = 0; m < in.size.m; m++ )
			for ( int i = 0; i < in.size.x; i++ )
				for ( int j = 0; j < in.size.y; j++ )
					for ( int z = 0; z < in.size.z; z++ )
						in_bin.data[in_bin(m, i, j, z)] = in(m, i, j, z) >= 0 ? 1 : 0;
					
    }

    void calculate_alpha()
    {
        // CAN USE MULTIPLE BINARISATION BY CHANGING `calculate_alpha` and `multiply_by_alpha` FUNCTIONS
        float sum = 0;

        for (int i = 0; i < weights.size.x; ++i)
            for (int j = 0; j < weights.size.y; j++)
                sum += weights(i, j, 0, 0);
        int n = weights.size.z * weights.size.y * weights.size.x;      // NO OF ELEMENTS IN `weights`
        alpha = sum / n;
        cout<<"*********mean********\n";
        cout<<alpha<<endl;
    }

    void multiply_by_alpha()
    {
		for ( int m = 0; m < out.size.m; m++ )
			for (int n = 0; n < out.size.x; n++)
				out(m, n, 0 ,0) = out(m, n, 0, 0) * alpha;

    }   

	int map( point_t d )
	// `tensor_t` SAVES DATA IN 1D FORMAT. `map` MAPS 3D POINT TO 1D TENSOR.
	{
		return d.m * (in.size.x * in.size.y * in.size.z) +
			d.z * (in.size.x * in.size.y) +
			d.y * (in.size.x) +
			d.x;
	}

	void activate()
 
	 //  `activate` FORWARD PROPOGATES AND SAVES THE RESULT IN `out` VARIABLE.

	{

        binarize();
        cout<<"**********binary weights for fc bin **********";
        print_tensor_bin(weights_bin);
        cout<<"**********binary inputs for fc bin ************";
        print_tensor_bin(in_bin);
        
        calculate_alpha();
		for( int e = 0; e < in.size.m; e++)
			for(int n = 0; n < out.size.x; n++ ){
				int inputv = 0;
				for ( int i = 0; i < in.size.x; i++ )
					for ( int j = 0; j < in.size.y; j++ )
						for ( int z = 0; z < in.size.z; z++ )
						{ 
							int m = map( { e, i, j, z } );
							inputv += !(in_bin.data[in_bin( e, i, j, z )] ^ weights_bin.data[weights_bin( e, m, n, 0 )]);
						}
				
				out( e, n, 0, 0 ) = 2*inputv - weights.size.x;   // 2P-N
				//cout<<out( n, 0, 0 )<<endl;
			}
		cout<<"********** output before multiplying*******";
		print_tensor(out);

        multiply_by_alpha();

		cout << "*************output of fc bin *************";
		print_tensor(out);
	}

	void fix_weights()
	{
		// for ( int n = 0; n < out.size.x; n++ )
		// {
		// 	gradient_t& grad = gradients[n];
		// 	for ( int i = 0; i < in.size.x; i++ )
		// 		for ( int j = 0; j < in.size.y; j++ )
		// 			for ( int z = 0; z < in.size.z; z++ )
		// 			{
		// 				int m = map( { i, j, z } );
		// 				float& w = weights( m, n, 0 );
		// 				w = update_weight( w, grad, in( i, j, z ) );
		// 			}

		// 	update_gradient( grad );
		// }
	}

	void calc_grads( tensor_t<float>& grad_next_layer )
	
	// CALCULATES BACKWARD PROPOGATION AND SAVES RESULT IN `grads_in`. 
	

	{
	// 	memset( grads_in.data, 0, grads_in.size.x *grads_in.size.y*grads_in.size.z * sizeof( float ) );
	// 	for ( int n = 0; n < out.size.x; n++ )
	// 	{
	// 		gradient_t& grad = gradients[n];
	// 		//grad.grad = grad_next_layer( n, 0, 0 ) * activator_derivative( input[n] );

	// 		for ( int i = 0; i < in.size.x; i++ )
	// 			for ( int j = 0; j < in.size.y; j++ )
	// 				for ( int z = 0; z < in.size.z; z++ )
	// 				{
	// 					int m = map( { i, j, z } );
	// 					grads_in( i, j, z ) += grad.grad * weights( m, n, 0 );
	// 				}
	// 	}
	}
};
#pragma pack(pop)
