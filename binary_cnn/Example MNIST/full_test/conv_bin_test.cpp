
#include<bits/stdc++.h>
#include "byteswap.h"
#include "../../CNN/cnn.h"

using namespace std;

// double train( vector<layer_t*>& layers, tensor_t<double>& data, tensor_t<double>& expected )
// {
// 	for ( int i = 0; i < layers.size(); i++ )
// 	{
// 		if ( i == 0 )
// 			activate( layers[i], data );
// 		else
// 			activate( layers[i], layers[i - 1]->out );
// 	}

// 	tensor_t<double> grads = layers.back()->out - expected;

// 	for ( int i = layers.size() - 1; i >= 0; i-- )
// 	{
// 		if ( i == layers.size() - 1 )
// 			calc_grads( layers[i], grads );
// 		else
// 			calc_grads( layers[i], layers[i + 1]->grads_in );
// 	}

// 	for ( int i = 0; i < layers.size(); i++ )
// 	{
// 		fix_weights( layers[i] );
// 	}

// 	double err = 0;
// 	for ( int i = 0; i < grads.size.x * grads.size.y * grads.size.z; i++ )
// 	{
// 		double f = expected.data[i];
// 		if ( f > 0.5 )
// 			err += abs(grads.data[i]);
// 	}
// 	return err * 100;
// }


// void forward( vector<layer_t*>& layers, tensor_t<double>& data )
// {
// 	for ( int i = 0; i < layers.size(); i++ )
// 	{
// 		if ( i == 0 )
// 			activate( layers[i], data );
// 		else
// 			activate( layers[i], layers[i - 1]->out );
// 	}
// }

// struct case_t
// {
// 	tensor_t<double> data;
// 	tensor_t<double> out;
// };

// uint8_t* read_file( const char* szFile )
// {
// 	//cout<<szFile;
	
// 	ifstream file( szFile, ios::binary | ios::ate );
// 	streamsize size = file.tellg();
// 	//cout<<size;
// 	file.seekg( 0, ios::beg );

// 	if ( size == -1 )
// 		return nullptr;

// 	uint8_t* buffer = new uint8_t[size];
// 	file.read( (char*)buffer, size );
// 	//cout<<(*buffer)<<endl;
// 	return buffer;
// }

// vector<case_t> read_test_cases()
// {
// 	vector<case_t> cases;
	
// 	//cout<<"***********"<<endl;
	
// 	uint8_t* train_image = read_file( "../../train-images.idx3-ubyte" );
// 	uint8_t* train_labels = read_file( "../../train-labels.idx1-ubyte" );
	
// 	//cout<<(*train_image)<<endl;
	
// 	uint32_t case_count = byteswap_uint32( *(uint32_t*)(train_image + 4) );

// 	cout<<case_count<<endl;
	
// 	for ( int i = 0; i < case_count; i++ )
// 	{
// 		case_t c {tensor_t<double>( 28, 28, 1 ), tensor_t<double>( 10, 1, 1 )};

// 		uint8_t* img = train_image + 16 + i * (28 * 28);

// 		uint8_t* label = train_labels + 8 + i;

// 		for ( int x = 0; x < 28; x++ )
// 			for ( int y = 0; y < 28; y++ )
// 				{c.data( x, y, 0 ) = img[x + y * 28] / 255.f;
// 					//cout<<(int)img[x+y*28]<<endl;
// 					}

// 		for ( int b = 0; b < 10; b++ )
// 			c.out( b, 0, 0 ) = *label == b ? 1.0f : 0.0f;

// 		cases.push_back( c );
// 	}
// 	delete[] train_image;
// 	delete[] train_labels;

// 	return cases;
// }

int main()
{
	//cout<<"kunal"<<endl;
	//vector<case_t> cases = read_test_cases();
	
	vector<layer_t*> layers;
	
	tensor_t<double> temp_in(1, 5,5,1);
	
	
	
	for(int k=0; k<1; k++)
	for(int i=0; i<5; i++){
		for(int j=0; j<5; j++){
			temp_in(0, i,j,k) = pow(-1,i^j)*2+i+j-4;
		}
	}
	
	tensor_t<double> t_grads(1,3,3,1);
	for(int i=0; i<3; i++){
		for(int j=0; j<3; j++){
			t_grads(0, i,j,0) = pow(-1,i^j)*2+i+j-4;
		}
	}
	//debug
	cout<<"*********input image *******"<<endl;
	print_tensor(temp_in);

	// conv_layer_t * layer1 = new conv_layer_t(1, 3, 1, temp_in.size);		// 28 * 28 * 1 -> 24 * 24 * 8
	// layer1->activate(temp_in);
	// prelu_layer_t * layer2 = new prelu_layer_t( {1,3,3,1} );
	// layer2->activate(layer1->out);
	conv_layer_bin_t * layer3 = new conv_layer_bin_t(1, 3, 1, {1,5,5,1});
	layer3->activate(temp_in);

	cout<<"******grads_next_layer*******\n";
	print_tensor(t_grads);
	layer3->calc_grads(t_grads);
	// fc_layer_t * layer4 = new fc_layer_t({1,2,2,1}, 3);
	// layer4->activate(layer3->out);
	// fc_layer_bin_t * layer5 = new fc_layer_bin_t({1,3,1,1}, 2);
	// layer5->activate(layer4->out);
	// scale_layer_t * layer6 = new scale_layer_t({1,2,1,1});
	// layer6->activate(layer5->out);
	// softmax_layer_t * layer7 = new softmax_layer_t({1,2,1,1});
	// layer7->activate(layer6->out);
	
	 //cout<<"******** conv"
	//layer1->activate(temp_in);
	//layers.push_back( (layer_t*)layer1 );
	
	



	//~ double amse = 0;
	//~ int ic = 0;

	//~ for ( long ep = 0; ep < 100000; ep++)
	//~ {
//~ //cout<<ep<<endl;
		//~ for ( auto t : cases )
		//~ {
			//~ double xerr = train( layers, t.data, t.out );
			//~ amse += xerr;

			//~ ep++;
			//~ ic++;

			//~ if ( ep % 1000 == 0 )
				//~ cout << "case " << ep << " err=" << amse/ic << endl;
			
			//~ //if(ep==1000) break;
			
			//~ // if ( GetAsyncKeyState( VK_F1 ) & 0x8000 )
			//~ // {
			//~ //	   printf( "err=%.4f%\n", amse / ic  );
			//~ //	   goto end;
			//~ // }
		//~ }
		//~ //xxs:;
		//~ //cout<<"kjsifsn";
		//~ //break;
	//~ }
	//~ // end:
//~ //xxs:;
cout<<"***************";

// 	while ( false )
// 	{
// 		uint8_t * data = read_file( "../test.ppm" );

// 		if ( data )
// 		{
// 			uint8_t * usable = data;

// 			while ( *(uint32_t*)usable != 0x0A353532 )
// 				usable++;

// #pragma pack(push, 1)
// 			struct RGB
// 			{
// 				uint8_t r, g, b;
// 			};
// #pragma pack(pop)

// 			RGB * rgb = (RGB*)usable;

// 			tensor_t<double> image(28, 28, 1);
// 			for ( int i = 0; i < 28; i++ )
// 			{
// 				for ( int j = 0; j < 28; j++ )
// 				{
// 					RGB rgb_ij = rgb[i * 28 + j];
// 					image( j, i, 0 ) = (((double)rgb_ij.r
// 							     + rgb_ij.g
// 							     + rgb_ij.b)
// 							    / (3.0f*255.f));
// 				}
// 			}

// 			forward( layers, image );
// 			tensor_t<double>& out = layers.back()->out;
// 			cout<<"&&&&&&&&&";
// 			for ( int i = 0; i < 10; i++ )
// 			{
// 				printf( "[%i] %f\n", i, out( i, 0, 0 )*100.0f );
// 			}

// 			delete[] data;
// 		}

// 		struct timespec wait;
// 		wait.tv_sec = 1;
// 		wait.tv_nsec = 0;
// 		nanosleep(&wait, nullptr);
// 	}
	return 0;
}
