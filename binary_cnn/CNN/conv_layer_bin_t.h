#pragma once
#include "layer_t.h"

#pragma pack(push, 1)
struct conv_layer_bin_t
{
	layer_type type = layer_type::conv_bin;
	tensor_t<float> grads_in;
	tensor_t<float> in;
	tensor_t<float> out;
	tensor_bin_t in_bin;
	tensor_bin_t in_bin2;
	tensor_bin_t out_bin;
	tensor_t<float> filters; //std::vector<tensor_t<float>> filters;
	tensor_bin_t filters_bin; //vector<tensor_bin_t> filters_bin;
	tensor_t<gradient_t> filter_grads; //std::vector<tensor_t<gradient_t>> filter_grads;
	uint16_t stride;
	uint16_t extend_filter;
	vector<float> alpha;
	vector<float> alpha2;
	
	conv_layer_bin_t( uint16_t stride, uint16_t extend_filter, uint16_t number_filters, tdsize in_size )
		:
		grads_in(in_size.m, in_size.x, in_size.y, in_size.z),
		in(in_size.m, in_size.x, in_size.y, in_size.z ),
		out(in_size.m,
		(in_size.x - extend_filter) / stride + 1,
			(in_size.y - extend_filter) / stride + 1,
			number_filters
		),
		in_bin(in_size.m, in_size.x, in_size.y, in_size.z),
		in_bin2(in_size.m, in_size.x, in_size.y, in_size.z),
		out_bin(in_size.m,
		(in_size.x - extend_filter) / stride + 1,
			(in_size.y - extend_filter) / stride + 1,
			number_filters
		),
		filters(number_filters, extend_filter, extend_filter, in_size.z),
		filter_grads(number_filters, extend_filter, extend_filter, in_size.z),
		filters_bin(number_filters, extend_filter, extend_filter, in_size.z)

	{
		this->stride = stride;
		this->extend_filter = extend_filter;
		assert( (float( in_size.x - extend_filter ) / stride + 1)
				==
				((in_size.x - extend_filter) / stride + 1) );

		assert( (float( in_size.y - extend_filter ) / stride + 1)
				==
				((in_size.y - extend_filter) / stride + 1) );

		for ( int a = 0; a < number_filters; a++ )
		{
			// tensor_t<float> t( extend_filter, extend_filter, in_size.z );
			// tensor_bin_t tb(extend_filter, extend_filter, in_size.z);
			int maxval = extend_filter * extend_filter * in_size.z;

			for ( int i = 0; i < extend_filter; i++ ){
			
				for ( int j = 0; j < extend_filter; j++ ){
					
					for ( int z = 0; z < in_size.z; z++ ){
						 //initialization of floating weights 
						filters(a,i, j, z ) =  (1.0f * (rand()-rand())) / float( RAND_MAX );
						
						// initialization of binary weights
						filters_bin.data[filters_bin(a,i,j,z)] = 0;
					}
				}
			}
			// filters.push_back( t );
			// filters_bin.push_back(tb);
		}

		cout<<"******weights for conv_bin********\n"<<endl;
		print_tensor(filters);
		
		// for ( int i = 0; i < number_filters; i++ )
		// {
		// 	tensor_t<gradient_t> t( extend_filter, extend_filter, in_size.z );
		// 	filter_grads.push_back( t );
		// }

	}

	point_t map_to_input( point_t out, int z )
	{
		out.x *= stride;
		out.y *= stride;
		out.z = z;
		return out;
	}

	struct range_t
	{
		int min_x, min_y, min_z;
		int max_x, max_y, max_z;
	};

	int normalize_range( float f, int max, bool lim_min )
	{
		if ( f <= 0 )
			return 0;
		max -= 1;
		if ( f >= max )
			return max;

		if ( lim_min ) // left side of inequality
			return ceil( f );
		else
			return floor( f );
	}

	range_t map_to_output( int x, int y )
	{
		float a = x;
		float b = y;
		return
		{
			normalize_range( (a - extend_filter + 1) / stride, out.size.x, true ),
			normalize_range( (b - extend_filter + 1) / stride, out.size.y, true ),
			0,
			normalize_range( a / stride, out.size.x, false ),
			normalize_range( b / stride, out.size.y, false ),
			(int)filters.size.m - 1,
		};
	}

	void activate( tensor_t<float>& in )
	{
		this->in = in;
		activate();
	}
	
	
	void binarize(){
		
		// binarizes weights
		for(int filter = 0; filter<filters.size.m; filter++){
			
			// tensor_t<float> &tf = filters[filter];
			// tensor_bin_t &tb = filters_bin[filter];
			
			for(int x=0; x< filters.size.x; x++){
				for(int y=0; y< filters.size.y; y++){
					for(int z=0; z< filters.size.z; z++){
						// ************************ remember always take var.data[var(x,y,z)] *****************************
						if(filters(filter,x,y,z) >= 0) filters_bin.data[filters_bin(filter,x,y,z)] = 1;
						else filters_bin.data[filters_bin(filter,x,y,z)] = 0;
					}
				}
			}
		}

		cout<<"************ binarized weights **********\n";
		print_tensor_bin(filters_bin);
		
		//binarizes in 
		for(int example = 0; example<in.size.m; example++)
			for(int x=0; x<in.size.x; x++){
				for(int y=0; y<in.size.y; y++){
					for(int z=0; z<in.size.z; z++){
						in_bin.data[in_bin(example,x,y,z)] = in(example,x,y,z)>=0 ? 1 : 0;
						
					}
				}
			}
		cout<<"******binarize input**************"<<endl;
		print_tensor_bin(in_bin);
		
	}
	
	void cal_mean(){
		//cout<<filters.size()<<' '<<filters[0].size.x<<' '<<filters[0].size.y<<' '<<filters[0].size.z<<endl;
		
		alpha.resize(in.size.m);
		
		for(int e = 0; e<in.size.m; e++){
			float sum = 0;
			// tensor_t<float> &tf = filters[filter];
			for(int x=0; x<in.size.x; x++)
				for(int y=0; y<in.size.y; y++)
					for(int z=0; z<in.size.z; z++){
						sum += abs(in(e,x,y,z));
				}
			
			alpha[e] = sum/(in.size.x*in.size.y*in.size.z);
			cout<<"alpha "<<endl;
			cout<<"alpha for"<<e<<"th example is "<<alpha[e]<<endl;
		}

		alpha2.resize(in.size.m);
		tensor_t<float> temp(in.size.m, in.size.x, in.size.y, in.size.z);

		for(int e = 0; e<in.size.m; e++){
			float sum = 0;
			// tensor_t<float> &tf = filters[filter];
			for(int x=0; x<in.size.x; x++)
				for(int y=0; y<in.size.y; y++)
					for(int z=0; z<in.size.z; z++){
						temp(e,x,y,z) = in(e,x,y,z) - alpha[e]*(in_bin(e,x,y,z));
						in_bin2.data[in_bin2(e,x,y,z)] = temp(e,x,y,z)>=0? 1 : 0;
						sum += abs(temp(e,x,y,z));
				}
			alpha2[e] = sum/(in.size.x*in.size.y*in.size.z);

			cout<<"alpha2"<<endl;
			cout<<"alpha2 for "<<e<<"th example is "<<alpha2[e]<<endl;
		}


		//~ return sum/(filters.size()*);
	}

	
	void activate()
	{
		//binarize filters and in 
		binarize();

		//initialize alpha for xornet
		cal_mean();

		//binarize convolution :)

		for(int example = 0; example<in.size.m; example++){
		
			for ( int filter = 0; filter < filters_bin.size.m; filter++ )
			{
				// tensor_bin_t &filter_data = filters_bin[filter];
				for ( int x = 0; x < out.size.x; x++ )
				{
					for ( int y = 0; y < out.size.y; y++ )
					{
						point_t mapped = map_to_input( { (uint16_t)x, (uint16_t)y, 0 }, 0 );
						float sum = 0, sum2 = 0;
						for ( int i = 0; i < extend_filter; i++ )
							for ( int j = 0; j < extend_filter; j++ )
								for ( int z = 0; z < in.size.z; z++ )
								{
									bool f = filters_bin.data[filters_bin(filter, i, j, z )];
									bool v = in_bin.data[in_bin(example, mapped.x + i, mapped.y + j, z )];
									bool v2 = in_bin2.data[in_bin2(example, mapped.x + i, mapped.y + j, z)];
									sum += (!(f^v));
									sum2 += (!(f^v2));
								}
						out(example, x, y, filter ) = alpha[example]*(2*sum - extend_filter*extend_filter*in.size.z);
						
						out(example, x,y,filter) += alpha2[example]*(2*sum2 - extend_filter*extend_filter*in.size.z);
						
					}
				}
			}
		}	


		cout<<"*********output for conv_bin*********\n";
		print_tensor(out);

		// conv_layer_t temp_conv(1, this->extend_filter, filters.size.m, in.size);
		// temp_conv.filters = this->filters;

		// cout<<"*******output if weights and input is float*******\n";
		// temp_conv.activate(this->in);

	}
	
	
	
	void fix_weights(){
	// {
	// 	for ( int a = 0; a < filters.size(); a++ )
	// 		for ( int i = 0; i < extend_filter; i++ )
	// 			for ( int j = 0; j < extend_filter; j++ )
	// 				for ( int z = 0; z < in.size.z; z++ )
	// 				{
	// 					float& w = filters[a].get( i, j, z );
	// 					gradient_t& grad = filter_grads[a].get( i, j, z );
	// 					w = update_weight( w, grad );
	// 					update_gradient( grad );
	// 				}
	}

	void calc_grads( tensor_t<float>& grad_next_layer)
	{
		for(int m=0; m<in.size.m;m++){
		for(int e=0; e<filters.size.m; e++){

				for(int k=0; k<filters.size.z; k++){
					for(int j=0; j<filters.size.x; j++){
						for(int i=0; i<filters.size.y;i++){
							for(int l=j; l<(in.size.x - filters.size.x)+j; l++){
								for(int n=i; n<(in.size.y - filters.size.y)+i; n++){
									// filter_grads(e, j, i, k) += grad_next_layer(e, l, n ,k)*in(m, l, n, k);

								}

							}
						}
					}
				}
		}
	}
	}
};
#pragma pack(pop)
