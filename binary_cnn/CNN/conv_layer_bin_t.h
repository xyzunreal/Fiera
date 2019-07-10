#pragma once
#include "layer_t.h"
#include <climits>

typedef unsigned int uint128_t __attribute__((mode(TI)));

struct packed_var{
	tensor_t<uint64_t> packed_input, packed_weight;

	void operator = (packed_var t){
		this->packed_input = t.packed_input;
		this->packed_input = t.packed_input;
	}
};

#pragma pack(push, 1)
struct conv_layer_bin_t
{
	layer_type type = layer_type::conv_bin;
	tensor_t<float> in;
	tensor_t<float> al_b; 			// α1 * h1 + α2 * h2
	tensor_t<float> filters; 
	tensor_bin_t filters_bin; 
	tensor_t<gradient_t> filter_grads;
	tensor_t<uint64_t> packed_input, packed_weight;
	uint16_t stride;
	uint16_t extend_filter, number_filters;
	tdsize in_size, out_size;
	vector<float> alpha;
	vector<float> alpha2;
	bool debug,clip_gradients_flag; 	
	conv_layer_bin_t( uint16_t stride, uint16_t extend_filter, uint16_t number_filters, tdsize in_size, bool clip_gradients_flag = true, bool debug_flag = false)
		:
		filters(number_filters, extend_filter, extend_filter, in_size.z),
		filter_grads(number_filters, extend_filter, extend_filter, in_size.z),
		filters_bin(number_filters, extend_filter, extend_filter, in_size.z)

	{
		this->number_filters = number_filters;
		this->out_size =  {in_size.m, (in_size.x - extend_filter) / stride + 1, (in_size.y - extend_filter) / stride + 1, number_filters};
		this->in_size = in_size;
		this->debug = debug_flag;
		this->stride = stride;
		this->extend_filter = extend_filter;
		this->clip_gradients_flag = clip_gradients_flag;
		assert( (float( in_size.x - extend_filter ) / stride + 1)
				==
				((in_size.x - extend_filter) / stride + 1) );

		assert( (float( in_size.y - extend_filter ) / stride + 1)
				==
				((in_size.y - extend_filter) / stride + 1) );

		for ( int a = 0; a < number_filters; a++ ){
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
		}
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
			normalize_range( (a - extend_filter + 1) / stride, out_size.x, true ),
			normalize_range( (b - extend_filter + 1) / stride, out_size.y, true ),
			0,
			normalize_range( a / stride, out_size.x, false ),
			normalize_range( b / stride, out_size.y, false ),
			(int)filters.size.m - 1,
		};
	}

	void bitpack_64(tensor_t<float> in){

		// assert(in.size.z % 64 == 0);
			// packed_var pack;

			packed_input.resize({in.size.m, in.size.x, in.size.y, in.size.z/64});
			packed_weight.resize({filters.size.m, filters.size.x, filters.size.y, filters.size.z/64});

			for(int i=0; i<in.size.m; i++){
				for(int j=0; j<in.size.x; j++){
					for(int k=0; k<in.size.y; k++){
						for(int z=0; z<in.size.z; z+=64){
							
							const size_t UNIT_LEN = 64;
							std::bitset<UNIT_LEN> bits;

							for(int zz = z; zz<z+64; zz++)
								bits[zz-z] = in(i,j,k,zz) >= 0;

								static_assert(sizeof(decltype(bits.to_ullong())) * CHAR_BIT == 64,
									"bits.to_ullong() must return a 64-bit element");
								packed_input(i,j,k,z/64) = bits.to_ullong();
						}
					}
				}
			}
			
			for(int i=0; i<filters.size.m; i++){
				for(int j=0; j<filters.size.x; j++){
					for(int k=0; k<filters.size.y; k++){
						for(int z=0; z<filters.size.z; z+=64){
							
							const size_t UNIT_LEN = 64;
							std::bitset<UNIT_LEN> bits;

							for(int zz = z; zz<z+64; zz++)
								bits[zz-z] = filters(i,j,k,zz) >= 0;
								// cout<<bits[zz-z]<<' '/

								static_assert(sizeof(decltype(bits.to_ullong())) * CHAR_BIT == 64,
										"bits.to_ullong() must return a 64-bit element");
								packed_weight(i,j,k,z/64) = bits.to_ullong();
						}
					}
				}
			}

			// return pack;
	}

    
    // packed_var<uint128_t> bitpack_128(tensor_t<float> in){
    //     	packed_var<uint128_t> pack;

	// 		pack.packed_input.resize({in.size.m, in.size.x, in.size.y, in.size.z/128});
	// 		pack.packed_weight.resize({filters.size.m, filters.size.x, filters.size.y, filters.size.z/128});

	// 		for(int i=0; i<in.size.m; i++){
	// 			for(int j=0; j<in.size.x; j++){
	// 				for(int k=0; k<in.size.y; k++){
	// 					for(int z=0; z<in.size.z; z+=128){
							
	// 						const size_t UNIT_LEN = 128;
	// 						std::bitset<UNIT_LEN> bits;

	// 						for(int zz = z; zz<z+128; zz++)
	// 							bits[zz-z] = in(i,j,k,zz) >= 0;

	// 							static_assert(sizeof(decltype(bits.to_ullong())) * CHAR_BIT == 128,
	// 								"bits.to_ullong() must return a 64-bit element");
	// 							pack.packed_input(i,j,k,z/128) = bits.to_ullong();
	// 					}
	// 				}
	// 			}
	// 		}
			
	// 		for(int i=0; i<filters.size.m; i++){
	// 			for(int j=0; j<filters.size.x; j++){
	// 				for(int k=0; k<filters.size.y; k++){
	// 					for(int z=0; z<filters.size.z; z+=128){
							
	// 						const size_t UNIT_LEN = 128;
	// 						std::bitset<UNIT_LEN> bits;

	// 						for(int zz = z; zz<z+128; zz++)
	// 							bits[zz-z] = filters(i,j,k,zz) >= 0;
	// 							// cout<<bits[zz-z]<<' '/

	// 							static_assert(sizeof(decltype(bits.to_ullong())) * CHAR_BIT == 128,
	// 									"bits.to_ullong() must return a 64-bit element");
	// 							pack.packed_weight(i,j,k,z/128) = bits.to_ullong();
	// 					}
	// 				}
	// 			}
	// 		}
	// 		return pack;	

    // }

	tensor_bin_t binarize(tensor_t<float> in){
		
		tensor_bin_t in_bin(in.size.m, in_size.x, in_size.y, in_size.z );

		for(int filter = 0; filter<filters.size.m; filter++){
			for(int x=0; x< filters.size.x; x++){
				for(int y=0; y< filters.size.y; y++){
					for(int z=0; z< filters.size.z; z++){
						if(filters(filter,x,y,z) >= 0) filters_bin.data[filters_bin(filter,x,y,z)] = 1;
						else filters_bin.data[filters_bin(filter,x,y,z)] = 0;
					}
				}
			}
		}
	
		//binarizes in 
		for(int example = 0; example<in.size.m; example++)
			for(int x=0; x<in.size.x; x++)
				for(int y=0; y<in.size.y; y++)
					for(int z=0; z<in.size.z; z++)
						in_bin.data[in_bin(example,x,y,z)] = in(example,x,y,z)>=0 ? 1 : 0;
		
		return in_bin;
		
	}
	
	tensor_bin_t calculate_alpha(tensor_t<float> in, tensor_bin_t in_bin ){
	
		alpha.resize(in.size.m);
		alpha2.resize(in.size.m);

		// CALCULATE alpha1		
		for(int e = 0; e<in.size.m; e++)
		{
			float sum = 0;
			for(int x=0; x<in.size.x; x++)
				for(int y=0; y<in.size.y; y++)
					for(int z=0; z<in.size.z; z++)
						sum += abs(in(e,x,y,z));
			
			alpha[e] = sum/(in.size.x*in.size.y*in.size.z);

		}

		// CALCULATE alpha2

		tensor_bin_t in_bin2(in.size.m, in_size.x, in_size.y, in_size.z);
		tensor_t<float> temp(in.size.m, in.size.x, in.size.y, in.size.z);

		for(int e = 0; e<in.size.m; e++){
			float sum = 0;
			
			for(int x=0; x<in.size.x; x++)
				for(int y=0; y<in.size.y; y++)
					for(int z=0; z<in.size.z; z++){
						temp(e,x,y,z) = in(e,x,y,z) - alpha[e]*(in_bin.data[in_bin(e,x,y,z)]==1 ? float(1) : float(-1) );
						in_bin2.data[in_bin2(e,x,y,z)] = temp(e,x,y,z)>=0? 1 : 0;
						sum += abs(temp(e,x,y,z));
					}





			alpha2[e] = sum/(in.size.x*in.size.y*in.size.z);

		}

		return in_bin2;
	}
		
	tensor_t<float> calculate_al_b(tensor_bin_t in_bin, tensor_bin_t in_bin2){
		// CALCULATE al_b

		tensor_t<float> al_b(in_bin.size.m, in_size.x, in_size.y, in_size.z);
        // #pragma omp parallel for private(e, x, y, z,al_b) num_threads(25)
		for (int e = 0; e < in_bin.size.m; e++)
			for(int x=0; x<in_bin.size.x; x++)
				for(int y=0; y<in_bin.size.y; y++)
					for(int z=0; z<in_bin.size.z; z++){



						al_b(e, x, y, z) = (in_bin.data[in_bin(e, x, y, z)] == 1 ? float(1) : float(-1) );
								// alpha2[e] * (in_bin2(e, x, y, z) == 1 ? float(1) : float(-1) );
					}
		
		return al_b;
	}

	tensor_t<float> calculate_al_b_first_bn(tensor_t<float> in){

		tensor_t<float> al_b(in.size.m, in_size.x, in_size.y, in_size.z);
        
		#pragma omp parallel for private(e, x, y, z,al_b) num_threads(25)
		
		for (int e = 0; e < in.size.m; e++)
			for(int x=0; x<in.size.x; x++)
				for(int y=0; y<in.size.y; y++)
					for(int z=0; z<in.size.z; z++)
						al_b(e, x, y, z) = (in(e,x,y,z) >= 0.0 ? float(1) : float(-1) );
					
		return al_b;
	}
	
	tensor_t<float> activate_old(tensor_t<float> in, bool train = false)
	{
		auto start = std::chrono::high_resolution_clock::now();

		if (train) this->in = in;
		
		tensor_t<float> out(in.size.m, (in_size.x - extend_filter) / stride + 1, (in_size.y - extend_filter) / stride + 1, number_filters );
		
		//binarize filters and in 
        tensor_bin_t in_bin = binarize(in);
		
		//initialize alpha and calculate in_bin2
        tensor_bin_t in_bin2 = calculate_alpha(in, in_bin);
		
		tensor_t<float> al_b = 	calculate_al_b(in_bin, in_bin2);

		if(train) this->al_b = al_b;
		
		//calculating binary convolution
        // #pragma omp parallel for private(example, filter,x, y,i,sum) num_threads(25)
		for(int example = 0; example<in.size.m; example++)
			for ( int filter = 0; filter < filters_bin.size.m; filter++ )
				for ( int x = 0; x < out.size.x; x++ )
					for ( int y = 0; y < out.size.y; y++ ){

						point_t mapped = map_to_input( { 0, (uint16_t)x, (uint16_t)y, 0 }, 0 );
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
						out(example, x, y, filter ) = (2*sum - extend_filter*extend_filter*in.size.z);
						
						// out(example, x,y,filter) += alpha2[example]*(2*sum2 - extend_filter*extend_filter*in.size.z);
						
					}
			
			auto finish = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = finish - start;
		std::cout << "Old binarized time: " << elapsed.count() << " s\n";

		return out;

	}	


	tensor_t<float> activate(tensor_t<float> in, bool train = false){
		
		this->al_b = calculate_al_b_first_bn(in);
		tensor_t<float> out(in.size.m, (in_size.x - extend_filter) / stride + 1, (in_size.y - extend_filter) / stride + 1, number_filters );
		
		// auto start = std::chrono::high_resolution_clock::now();
		

		// if(in.size.z % 128==0){
		// 	packed_var<uint128_t> pack;
		// 	pack = bitpack_128(in, 128);

		// 	for(int example = 0; example<pack.packed_input.size.m; example++){
		// 		for ( int filter = 0; filter < pack.packed_weight.size.m; filter++ )
		// 			for ( int x = 0; x < out.size.x; x++ )
		// 				for ( int y = 0; y < out.size.y; y++ ){

		// 					point_t mapped = map_to_input( { 0, (uint16_t)x, (uint16_t)y, 0 }, 0 );
		// 					float sum = 0, sum2 = 0;
		// 					for ( int i = 0; i < extend_filter; i++ )
		// 						for ( int j = 0; j < extend_filter; j++ )
		// 							for ( int z = 0; z < pack.packed_input.size.z; z++ )
		// 							{
		// 								uint128_t xnor = ~(pack.packed_input(example,mapped.x + i, mapped.y + j, z)
		// 													^pack.packed_weight(filter, i, j, z));
		// 								sum += __builtin_popcount(xnor);
		// 							}
		// 					out(example, x, y, filter ) = (2*sum - extend_filter*extend_filter*in.size.z);
							
		// 				}
		// 	}

		// }
		assert(in.size.z % 64 == 0);

		bitpack_64(in);

		for(int example = 0; example<packed_input.size.m; example++){
			for ( int filter = 0; filter < packed_weight.size.m; filter++ )
				for ( int x = 0; x < out.size.x; x++ )
					for ( int y = 0; y < out.size.y; y++ ){

						point_t mapped = map_to_input( { 0, (uint16_t)x, (uint16_t)y, 0 }, 0 );
						float sum = 0, sum2 = 0;
						for ( int i = 0; i < extend_filter; i++ )
							for ( int j = 0; j < extend_filter; j++ )
								for ( int z = 0; z < packed_input.size.z; z++ )
								{
									uint64_t xnor = ~(packed_input(example,mapped.x + i, mapped.y + j, z)
														^ packed_weight(filter, i, j, z));
									sum += __builtin_popcount(xnor);
								}
						out(example, x, y, filter ) = (2*sum - extend_filter*extend_filter*in.size.z);
						
					}
		}
		
		// auto finish = std::chrono::high_resolution_clock::now();
		// std::chrono::duration<double> elapsed = finish - start;
		// std::cout << "New binarized time: " << elapsed.count() << " s\n";

		return out;

	}
	void fix_weights(float learning_rate){
		for ( int a = 0; a < filters.size.m; a++ )
			for ( int i = 0; i < extend_filter; i++ )
				for ( int j = 0; j < extend_filter; j++ )
					for ( int z = 0; z < in.size.z; z++ )
					{
						float& w = filters(a, i, j, z );
						gradient_t& grad = filter_grads(a, i, j, z );
						// grad.grad /= in.size.m;
						w = update_weight( w, grad, 1, true, learning_rate);
						update_gradient(grad);
					}
	}

	tensor_t<float> calc_grads( tensor_t<float>& grad_next_layer)
	{
		auto start = std::chrono::high_resolution_clock::now();

		tensor_t<float> grads_in(grad_next_layer.size.m, in_size.x, in_size.y, in_size.z);

		for ( int k = 0; k < filter_grads.size.m; k++ )
		{
			for ( int i = 0; i < extend_filter; i++ )
				for ( int j = 0; j < extend_filter; j++ )
					for ( int z = 0; z < in.size.z; z++ )
						filter_grads(k, i, j, z ).grad = 0;
		}

        // #pragma omp parallel for private(e, x, y,z,sum_error) num_threads(25)
		for ( int e = 0; e < in.size.m; e++)
			for ( int x = 0; x < in.size.x; x++ )
				for ( int y = 0; y < in.size.y; y++ ){
					range_t rn = map_to_output( x, y );
					for ( int z = 0; z < in.size.z; z++ ){
						
						float sum_error = 0;
						for ( int i = rn.min_x; i <= rn.max_x; i++ ){
							int minx = i * stride;
							for ( int j = rn.min_y; j <= rn.max_y; j++ ){
								int miny = j * stride;
								for ( int k = rn.min_z; k <= rn.max_z; k++ ){
									
									filter_grads(k, x - minx, y - miny, z ).grad += al_b(e, x, y, z ) * grad_next_layer(e, i, j, k );
									clip_gradients(clip_gradients_flag, filter_grads(k, x - minx, y - miny, z ).grad);
									if(fabs(in(e,x,y,z)) <= 1){
										float w_applied = (filters_bin.data[filters_bin(k, x - minx, y - miny, z )] == 1? float(1) : float(-1));
										sum_error += w_applied * grad_next_layer(e, i, j, k );
									}
									else sum_error = 0;
								}
							}
						}
						grads_in( e, x, y, z ) = sum_error;
						clip_gradients(clip_gradients_flag, grads_in(e,x,y,z));
					}
				}
		
		auto finish = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = finish - start;
		std::cout << "backprop time: " << elapsed.count() << " s\n";
		
		return grads_in;
	}

	void save_layer( json& model ){
		model["layers"].push_back( {
			{ "layer_type", "conv_bin" },
			{ "stride", stride },
			{ "extend_filter", extend_filter },
			{ "number_filters", filters.size.m },
			{ "in_size", {in_size.m, in_size.x, in_size.y, in_size.z} },
			{ "clip_gradients", clip_gradients_flag}
		} );
	}

	void save_layer_weight( string fileName ){
		ofstream file(fileName);
		int m = filters.size.m;
		int x = filters.size.x;
		int y = filters.size.y;
		int z = filters.size.z;
		int array_size = m*x*y*z;
		
		vector<float> data;
		for ( int i = 0; i < array_size; i++ )
			data.push_back(filters.data[i]);	
		json weights = { 
			{ "type", "conv" },
			{ "size", array_size },
			{ "data", data}
		};
		file << weights << endl;
		file.close();
	}

	void load_layer_weight( string fileName ){
		ifstream file(fileName);
		json weights;
		file >> weights;
		assert(weights["type"] == "conv");
		vector<float> data = weights["data"];
		int size  = weights["size"];
		for (int i = 0; i < size; i++)
			this->filters.data[i] = data[i];
		file.close();
	}


	void print_layer(){
		cout << "\n\n Conv Binary Layer : \t";
		cout << "\n\t in_size:\t";
		print_tensor_size(in_size);
		cout << "\n\t Filter Size:\t";
		print_tensor_size(filters.size);
		cout << "\n\t out_size:\t";
		print_tensor_size(out_size);
	}
};
#pragma pack(pop)
