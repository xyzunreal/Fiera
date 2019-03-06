#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "byteswap.h"
#include "../../CNN/cnn.h"

using namespace std;

int mini_batch_size = 6;
int num_batches = 6000/mini_batch_size;	
	
struct case_t
{
	tensor_t<double> data;
	tensor_t<double> out;
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

	case_t c {tensor_t<double>( mini_batch_size, 28, 28, 1 ), tensor_t<double>(mini_batch_size, 10, 1, 1 )};
 
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
    conv_layer_t * layer1 = new conv_layer_t(1, 3, 8, cases[0].data.size);		
    prelu_layer_t * layer2 = new prelu_layer_t( layer1->out.size);
    conv_layer_bin_t * layer3 = new conv_layer_bin_t(1, 3, 16, layer2->out.size);		
    prelu_layer_t * layer4 = new prelu_layer_t( layer3->out.size);
    conv_layer_bin_t * layer5 = new conv_layer_bin_t(1, 3, 16, layer4->out.size);		
    prelu_layer_t * layer6 = new prelu_layer_t( layer5->out.size);
    // conv_layer_bin_t * layer7 = new conv_layer_bin_t(1, 3, 16, layer6->out.size);		
    // prelu_layer_t * layer8 = new prelu_layer_t( layer7->out.size);
    // conv_layer_bin_t * layer9 = new conv_layer_bin_t(1, 3, 16, layer8->out.size);		
    // prelu_layer_t * layer10 = new prelu_layer_t( layer9->out.size);
    fc_layer_t * layer11 = new fc_layer_t(layer6->out.size,10);
    prelu_layer_t * layer12 = new prelu_layer_t( layer11->out.size); 
    fc_layer_bin_t * layer13 = new fc_layer_bin_t(layer12->out.size, 10);
    scale_layer_t * layerS = new scale_layer_t(layer13->out.size);
    softmax_layer_t * layer14 = new softmax_layer_t(layerS->out.size, false, false, false);

    vector<double> cost_vec;
    cost_vec.push_back(0);
    double learning_rate = 0.0001;


    for(int epoch = 0; epoch<400; epoch++){

        //batch_num<num_batches
        for(int batch_num = 0; batch_num<1; batch_num++){
                // cout<<"*************epoch number*********** "<<epoch<<"***********************\n";
                layer1->activate(cases[batch_num].data);
                // cout<<"*************epoch number*********** "<<epoch<<"***********************\n";
                layer2->activate(layer1->out);
                // cout<<"*************epoch number*********** "<<epoch<<"***********************\n";
                layer3->activate(layer2->out);
                // cout<<"*************epoch number*********** "<<epoch<<"***********************\n";
                layer4->activate(layer3->out);
                // cout<<"*************epoch number*********** "<<epoch<<"***********************\n";
                layer5->activate(layer4->out);
                // cout<<"*************epoch number*********** "<<epoch<<"***********************\n";
                layer6->activate(layer5->out);
                // layer7->activate(layer6->out);
                // layerS->activate(layer5->out);
                // layer8->activate(layer7->out);
                // layer9->activate(layer8->out);
                // cout<<"*************epoch number*********** "<<epoch<<"***********************\n";
                // layer10->activate(layer9->out);
                layer11->activate(layer6->out);
                // layerS->activate(layer5->out);
                layer12->activate(layer11->out);
                layer13->activate(layer12->out);
                layerS->activate(layer13->out);
                layer14->activate(layerS->out);
                // if (epoch>1)
                // {
                // cout << "layer6->out\n\n\n\n";
                // print_tensor(layer5->out);
                // }
                
                // tensor_t<double> costs = cross_entropy(layer6->out, cases[batch_num].out);
                // double costs_avg = 0;

                // for(int e = 0; e<mini_batch_size; e++){
                //     costs_avg += costs(e,0,0,0);
                // }

                // costs_avg /= (double)mini_batch_size;
                
                // cost_vec.push_back(costs_avg);
                

                double l1 = cross_entropy(layer14->out, cases[batch_num].out, false);
                double l2 = l1;
                // double l2 = cross_entropy(layer6->out, cases[batch_num].out)(1, 0, 0, 0);
                cout<<"loss for img1 ";
                cout<<l1<<endl;
                cout<<"loss for img 2";
                cout<<l2<<endl;
                
                cout<<"*****loss total ************\n";
                cout<<((l1+l2)/2)<<endl;

                cost_vec.push_back((l1+l2)/2);
                
                // cout<<"*************epoch number*********** "<<epoch<<"***********************\n";
                layer14->calc_grads(cases[batch_num].out);
                // cout<<"*************epoch number*********** "<<epoch<<"***********************\n";
                layerS->calc_grads(layer14->grads_in);
                layer13->calc_grads(layerS->grads_in);
                layer12->calc_grads(layer13->grads_in);
                // layerS->calc_grads(layer8->grads_in);
                layer11->calc_grads(layer12->grads_in);
                // cout<<"*************epoch number*********** "<<epoch<<"***********************\n";
                // layer10->calc_grads(layer11->grads_in);
                // cout<<"*************epoch number*********** "<<epoch<<"***********************\n";
                // layer9->calc_grads(layer10->grads_in);
                // cout<<"*************epoch number*********** "<<epoch<<"***********************\n";
                // layer8->calc_grads(layer9->grads_in);
                // cout<<"*************epoch number*********** "<<epoch<<"***********************\n";
                // layer7->calc_grads(layer8->grads_in);
                
                layer6->calc_grads(layer11->grads_in);
                layer5->calc_grads(layer6->grads_in);
                layer4->calc_grads(layer5->grads_in);
                layer3->calc_grads(layer4->grads_in);
                layer2->calc_grads(layer3->grads_in);
                layer1->calc_grads(layer2->grads_in);
                
                // double diff_cost = cost_vec[cost_vec.size()-1] - cost_vec[cost_vec.size()-2];
                
                // if ( diff_cost < 0.00001 )     // Stops training if cost decreases very slow
                //   break;
                // cout<<"*************epoch number*********** "<<epoch<<"***********************\n";
                layer1->fix_weights(learning_rate);
                // cout<<"*************epoch number*********** "<<epoch<<"***********************\n";
                layer2->fix_weights(learning_rate);
                // cout<<"*************epoch number*********** "<<epoch<<"***********************\n";
                layer3->fix_weights(learning_rate);
                // cout<<"*************epoch number*********** "<<epoch<<"***********************\n";
                layer4->fix_weights(learning_rate);
                // cout<<"*************epoch number*********** "<<epoch<<"***********************\n";
                layer5->fix_weights(learning_rate);
                // cout<<"*************epoch number*********** "<<epoch<<"***********************\n";
                layer6->fix_weights(learning_rate);
                // layer7->fix_weights(learning_rate);
                // layerS->fix_weights(learning_rate);
                // layer8->fix_weights(learning_rate);
                // cout<<"*************epoch number*********** "<<epoch<<"***********************\n";
                // layer9->fix_weights(learning_rate);
                // cout<<"*************epoch number*********** "<<epoch<<"***********************\n";
                // layer10->fix_weights(learning_rate);
                // cout<<"*************epoch number*********** "<<epoch<<"***********************\n";
                layer11->fix_weights(learning_rate);
                // cout<<"*************epoch number*********** "<<epoch<<"***********************\n";
                layer12->fix_weights(learning_rate);
                // cout<<"*************epoch number*********** "<<epoch<<"***********************\n";
                layer13->fix_weights(learning_rate);
                layerS->fix_weights(learning_rate);
         }
    }

    cout<<"**********cost*************\n";

    for(int i=0; i<cost_vec.size(); i++){
        cout<<cost_vec[i]<<endl;
    }


	return 0;
}
