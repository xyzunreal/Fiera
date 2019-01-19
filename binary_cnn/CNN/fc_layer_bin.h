#pragma once
#include <math.h>
#include <float.h>
#include <string.h>
#include "layer_t.h"
#include "optimization_method.h"
#include "gradient_t.h"
#include "tensor_bin_t.h"


using namespace std;

#pragma pack(push, 1)
struct fc_layer_bin_t
{
	layer_type type = layer_type::fc_bin;
	tensor_t<float> grads_in; 
	tensor_t<float> in;
	tensor_t<float> out;
    tensor_bin_t in_bin;    // 1st BINARIZATION
	tensor_bin_t in_bin2;	// 2nd BINARIZATION
	tensor_t<float> al_b;	
    tensor_bin_t out_bin;
    vector<float> alpha;
	vector<float> alpha2;

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
		in_bin2(in_size.m, in_size.x, in_size.y, in_size.z),
		al_b(in_size.m, in_size.x, in_size.y, in_size.z),
		out( in_size.m, out_size, 1, 1 ),
        out_bin( in_size.m, out_size, 1, 1),
        weights_bin(in_size.x * in_size.y * in_size.z, out_size, 1, 1 ),
		grads_in(in_size.m, in_size.x, in_size.y, in_size.z ),
		weights(in_size.x * in_size.y * in_size.z, out_size, 1, 1 ),
		// to be checked later :(
		gradients(in_size.x * in_size.y * in_size.z, out_size, 1, 1)
	{
		
		// WEIGHT INITIALIZATION
		
		for ( int i = 0; i < out_size; i++ )
			for ( int h = 0; h < in_size.x*in_size.y*in_size.z; h++ ){
				
				weights(h,i, 0, 0 ) =  (1.0f * (rand()-rand())) / float( RAND_MAX );
				weights_bin.data[weights_bin(h, i, 0, 0)] = 0;
			
			}
	
		cout<<"***********float weights for fc bin ************\n";
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
        for (int i = 0; i < weights.size.m; ++i){
            for (int j = 0; j < weights.size.x; j++){
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

	void calculate_alpha(){
		//cout<<filters.size()<<' '<<filters[0].size.x<<' '<<filters[0].size.y<<' '<<filters[0].size.z<<endl;
		
		alpha.resize(in.size.m);
		alpha2.resize(in.size.m);

		// CALCULATE alpha1		
		for(int e = 0; e < in.size.m; e++){
			float sum = 0;
			// tensor_t<float> &tf = filters[filter];
			for(int x = 0; x < in.size.x; x++)
				for(int y = 0; y < in.size.y; y++)
					for(int z = 0; z < in.size.z; z++){
						sum += abs(in(e,x,y,z));
				}
			
			alpha[e] = sum/(in.size.x*in.size.y*in.size.z);
			cout<<"alpha "<<endl;
			cout<<"alpha for"<<e<<"th example is "<<alpha[e]<<endl;
		}

		// CALCULATE alpha2
		tensor_t<float> temp(in.size.m, in.size.x, in.size.y, in.size.z);
		for(int e = 0; e<in.size.m; e++){
			float sum = 0;
			// tensor_t<float> &tf = filters[filter];
			for(int x=0; x<in.size.x; x++)
				for(int y=0; y<in.size.y; y++)
					for(int z=0; z<in.size.z; z++){
						temp(e,x,y,z) = in(e,x,y,z) - alpha[e]*(in_bin(e,x,y,z)==1 ? float(1) : float(-1) );
						in_bin2.data[in_bin2(e,x,y,z)] = temp(e,x,y,z)>=0? 1 : 0;
						sum += abs(temp(e,x,y,z));
				}
			alpha2[e] = sum/(in.size.x*in.size.y*in.size.z);

			cout << "******in_bin2******";
			print_tensor_bin(in_bin2);
			cout<<"alpha2"<<endl;
			cout<<"alpha2 for "<<e<<"th example is "<<alpha2[e]<<endl;
		}

		// CALCULATE al_b
		for (int e = 0; e < in.size.m; e++)
			for(int x=0; x<in.size.x; x++)
				for(int y=0; y<in.size.y; y++)
					for(int z=0; z<in.size.z; z++){
						al_b(e, x, y, z) = alpha[e] * (in_bin(e, x, y, z) == 1 ? float(1) : float(-1) ) +
								alpha2[e] * (in_bin2(e, x, y, z) == 1 ? float(1) : float(-1) );
					}
	
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
				int sum, sum2;
				for ( int i = 0; i < in.size.x; i++ )
					for ( int j = 0; j < in.size.y; j++ )
						for ( int z = 0; z < in.size.z; z++ )
						{
							int m = map( { 0 , i, j, z } );
							bool f = weights_bin.data[weights_bin( m, n, 0, 0 )];
							bool v = in_bin.data[in_bin(e, i, j, z)];
							bool v2 = in_bin2.data[in_bin2(e, i, j, z)];
							sum += !(f ^ v);
							sum2 += !(f ^ v2);
						}

				// weights.size.m is equals to total size of input. i.e. in.size.x * in.size.y * in.size.z
				out(e, n, 0, 0 ) = alpha[e] * ( 2 * sum - weights.size.m );			// alpha * ( 2P - N )
				out(e, n, 0, 0 ) += alpha2[e] * (2 * sum2 - weights.size.m ); 		// alpha2 * ( 2P - N )
			}
		cout<<"********** output before multiplying*******";
		print_tensor(out);


		cout << "*************output of fc bin *************";
		print_tensor(out);
	}

	void fix_weights()
	{
		for ( int n = 0; n < out.size.x; n++ )
		{
			for ( int i = 0; i < in.size.x; i++ )
				for ( int j = 0; j < in.size.y; j++ )
					for ( int z = 0; z < in.size.z; z++ )
					{
						int m = map( { 0, i, j, z } );
						float& w = weights( m, n, 0, 0 );
						gradient_t grad_sum;
						gradient_t weight_grad;
						for ( int e = 0; e < out.size.m; e++ ){			
							weight_grad = gradients(e, n, 0, 0) * al_b(e, i, j, z);	// d W = d A(l+1) * A(l)(binary) 
							grad_sum = weight_grad + grad_sum;
						}
						grad_sum = grad_sum / out.size.m;
						w = update_weight( w, grad_sum, 1, true); 
					}
			for (int e = 0; e < out.size.m; e++)
				update_gradient( gradients(e, n, 0, 0) );
		}
		cout<<"*******new weights for fc_bin*****\n";
		print_tensor(weights);
	}

	void calc_grads( tensor_t<float>& grad_next_layer )
	
	// CALCULATES BACKWARD PROPOGATION AND SAVES RESULT IN `grads_in`. 
	
	{
		memset( grads_in.data, 0, grads_in.size.x *grads_in.size.y*grads_in.size.z * sizeof( float ) );
		
		for(int e=0; e<in.size.m; e++)
			for ( int n = 0; n < out.size.x; n++ )
			{
				gradient_t& grad = gradients(e, n, 0, 0);
				grad.grad = grad_next_layer(e, n, 0, 0 );

				for ( int i = 0; i < in.size.x; i++ )
					for ( int j = 0; j < in.size.y; j++ )
						for ( int z = 0; z < in.size.z; z++ )
						{
							int m = map( {0, i, j, z } );
							if(in(e,i,j,z) <= 1)
								grads_in(e, i, j, z ) += grad.grad * (weights_bin.data[weights_bin( m, n,0, 0 )] == 1? float(1) : float(-1));
							else
								grads_in(e,i,j,z) += 0;
						}
			}

		// cout<<"*****grads_next_layer*****\n";
		// print_tensor(grad_next_layer);

		cout<<"grads_in for fc_bin ***********\n";
		print_tensor(grads_in);
			
	}
};
#pragma pack(pop)
