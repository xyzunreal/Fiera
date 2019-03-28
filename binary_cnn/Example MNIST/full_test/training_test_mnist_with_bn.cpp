#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "byteswap.h"
#include "../../CNN/cnn.h"

using namespace std;

int mini_batch_size = 64;
int num_batches = 6000/mini_batch_size;	
	
struct case_t
{
	tensor_t<float> data;
	tensor_t<float> out;
};

uint8_t* read_file( const char* szFile )
{
	ifstream file( szFile, ios::binary | ios::ate );
    streamsize size = file.tellg();
    
    // cout<<"size: "<<size<<endl;

	file.seekg( 0, ios::beg );

	if ( size == -1 )
		return nullptr;

	uint8_t* buffer = new uint8_t[size];
	file.read( (char*)buffer, size);
	// cout<<"buffer: "<<buffer[0]<<endl;
	return buffer;
}

vector<case_t> read_test_cases()
{
	vector<case_t> cases;
	
	uint8_t* train_image = read_file( "../../train-images.idx3-ubyte" );
	uint8_t* train_labels = read_file( "../../train-labels.idx1-ubyte" );
	
	// cout<<(train_image[0])<<endl;
	
	uint32_t case_count = byteswap_uint32( *(uint32_t*)(train_image + 4) );

	cout<<"case_count: "<<case_count<<endl;
	
    int crnt_count = 0;

	case_t c {tensor_t<float>( mini_batch_size, 28, 28, 1 ), tensor_t<float>(mini_batch_size, 10, 1, 1 )};
 
    for( int i = 0; i < case_count; i++ )
	{

		uint8_t* img = train_image + 16 + i * (28 * 28);

		uint8_t* label = train_labels + 8 + i;

		for ( int x = 0; x < 28; x++ )
			for ( int y = 0; y < 28; y++ ){
                    c.data(crnt_count, x, y, 0) = img[x + y * 28] / 255.f;
                }

		for ( int b = 0; b < 10; b++ )
			c.out(crnt_count , b, 0, 0 ) = *label == b ? 1.0f : 0.0f;

        crnt_count++;

        if(crnt_count == mini_batch_size){
		    cases.push_back(c);
            crnt_count = 0;
        }
	}

	delete[] train_image;
	delete[] train_labels;

	return cases;
}

int main()
{

	vector<case_t> cases = read_test_cases();
    print_tensor(cases[0].data);
    conv_layer_t * layer1 = new conv_layer_t(1, 3, 8, cases[0].data.size,false,false);		
    batch_norm_layer_t * layerbb = new batch_norm_layer_t(layer1->out.size,false,false);
    prelu_layer_t * layer2 = new prelu_layer_t( layerbb->out.size,false,false);
    conv_layer_t * layer3 = new conv_layer_t(1, 3, 16, layer2->out.size,false,false);		
    // batch_norm_layer_t * layerb = new batch_norm_layer_t(layer3->out.size,false,false);
    prelu_layer_t * layer4 = new prelu_layer_t( layer3->out.size,false,false);
    // fc_layer_t * layer5 = new fc_layer_t(layer4->out.size, 70,false,false);
    // prelu_layer_t * layer6 = new prelu_layer_t( layer4->out.size,false); 
    fc_layer_t * layer7 = new fc_layer_t(layer4->out.size, 10, false);
    // scale_layer_t * layerS = new scale_layer_t(layer5->out.size);
    softmax_layer_t * layer8 = new softmax_layer_t(layer7->out.size, false, false, false);

    vector<float> cost_vec;
    cost_vec.push_back(0);
    float learning_rate = 0.001;


    for(int epoch = 0; epoch<12; epoch++){

        //batch_num<num_batches
        for(int batch_num = 0; batch_num<10; batch_num++){
                // cout<<"*************epoch number*********** "<<epoch<<"***********************\n";
                layer1->activate(cases[batch_num].data);
                // cout<<"*************epoch number*********** "<<epoch<<"***********************\n";
                layerbb->activate(layer1->out);
                layer2->activate(layerbb->out);
                // cout<<"*************epoch number*********** "<<epoch<<"***********************\n";
                layer3->activate(layer2->out);
                // cout<<"*************epoch number*********** "<<epoch<<"***********************\n";
                // layerb->activate(layer3->out);
                layer4->activate(layer3->out);
                layer7->activate(layer4->out);
                // cout<<"*************epoch number*********** "<<epoch<<"***********************\n";
                // layer5->activate(layer4->out);
                // layer6->activate(layer4->out);
                // layerS->activate(layer5->out);
                cout<<"*************layer8*********** "<<endl;
                print_tensor(layer7->out);
                layer8->activate(layer7->out);
                
                
                // print_tensor(layerbb->out);

                float l1 = cross_entropy(layer8->out, cases[batch_num].out);
                float l2 = l1;
                // float l2 = cross_entropy(layer6->out, cases[batch_num].out)(1, 0, 0, 0);
                // cout<<"loss for img1 ";
                // cout<<l1<<endl;
                // cout<<"loss for img 2";
                // cout<<l2<<endl;
                
                cout<<"*****loss total ************\n";
                cout<<((l1+l2)/mini_batch_size)<<endl;

                cost_vec.push_back((l1+l2)/mini_batch_size);
                
                layer8->calc_grads(cases[batch_num].out);
                // cout<<"*************epoch number*********** "<<epoch<<"***********************\n";
                layer7->calc_grads(layer8->grads_in);
                // layer6->calc_grads(layer7->grads_in);
                // layerS->calc_grads(layer8->grads_in);
                // layer5->calc_grads(layer6->grads_in);
                // cout<<"*************epoch number*********** "<<epoch<<"***********************\n";
                layer4->calc_grads(layer7->grads_in);
                // layerb->calc_grads(layer4->grads_in);
                // cout<<"*************epoch number*********** "<<epoch<<"***********************\n";
                layer3->calc_grads(layer4->grads_in);
                // cout<<"*************epoch number*********** "<<epoch<<"***********************\n";
                layer2->calc_grads(layer3->grads_in);
                layerbb->calc_grads(layer2->grads_in);
                // cout<<"*************epoch number*********** "<<epoch<<"***********************\n";
                // cout<<"*************epoch number*********** "<<epoch<<"***********************\n";
                layer1->calc_grads(layerbb->grads_in);
                
                
                // float diff_cost = cost_vec[cost_vec.size()-1] - cost_vec[cost_vec.size()-2];
                
                // if ( diff_cost < 0.00001 )     // Stops training if cost decreases very slow
                //   break;
                // cout<<"*************epoch number*********** "<<epoch<<"***********************\n";
                layer1->fix_weights(learning_rate);
                // cout<<"*************epoch number*********** "<<epoch<<"***********************\n";
                layerbb->fix_weights(learning_rate);
                layer2->fix_weights(learning_rate);
                // cout<<"*************epoch number*********** "<<epoch<<"***********************\n";
                layer3->fix_weights(learning_rate);
                // cout<<"*************epoch number*********** "<<epoch<<"***********************\n";
                // layerb->fix_weights(learning_rate);
                layer4->fix_weights(learning_rate);
                // cout<<"*************epoch number*********** "<<epoch<<"***********************\n";
                // layer5->fix_weights(learning_rate);
                // cout<<"*************epoch number*********** "<<epoch<<"***********************\n";
                // layer6->fix_weights(learning_rate);
                layer7->fix_weights(learning_rate);
                // layerS->fix_weights(learning_rate);
                layer8->fix_weights(learning_rate);
        }
    }

    cout<<"**********cost*************\n";

    for(int i=0; i<cost_vec.size(); i++){
        cout<<cost_vec[i]<<endl;
    }


	return 0;
}
