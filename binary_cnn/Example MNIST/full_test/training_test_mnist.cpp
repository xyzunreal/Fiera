#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "byteswap.h"
#include "../../CNN/cnn.h"

using namespace std;

int mini_batch_size = 1;
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
    conv_layer_t * layer1 = new conv_layer_t(1, 3, 8, cases[0].data.size,true);		
    prelu_layer_t * layer2 = new prelu_layer_t( layer1->out.size,true);
    conv_layer_t * layer3 = new conv_layer_t(1, 3, 16, layer2->out.size,true);		
    prelu_layer_t * layer4 = new prelu_layer_t( layer3->out.size,true);
    fc_layer_t * layer5 = new fc_layer_t(layer4->out.size, 10, true);
    softmax_layer_t * layer6 = new softmax_layer_t(layer5->out.size, true,true);

    vector<float> cost_vec;
    cost_vec.push_back(0);
    float learning_rate = 0.001;


    for(int epoch = 0; epoch<1; epoch++){

        //batch_num<num_batches
        for(int batch_num = 0; batch_num<1; batch_num++){
                
                layer1->activate(cases[batch_num].data);
                layer2->activate(layer1->out);
                layer3->activate(layer2->out);
                layer4->activate(layer3->out);
                layer5->activate(layer4->out);
                layer6->activate(layer5->out);
                // if (epoch>1)
                // {
                // cout << "layer6->out\n\n\n\n";
                // print_tensor(layer5->out);
                // }
                
                // tensor_t<float> costs = cross_entropy(layer6->out, cases[batch_num].out);
                // float costs_avg = 0;

                // for(int e = 0; e<mini_batch_size; e++){
                //     costs_avg += costs(e,0,0,0);
                // }

                // costs_avg /= (float)mini_batch_size;
                
                // cost_vec.push_back(costs_avg);
                

                float l1 = cross_entropy(layer6->out, cases[batch_num].out)(0, 0, 0, 0);
                float l2 = l1;
                // cout<<"loss for img1 ";
                // cout<<l1<<endl;
                // cout<<"loss for img 2";
                // cout<<l2<<endl;
                
                cout<<"*****loss total ************\n";
                cout<<((l1+l2)/2)<<endl;

                cost_vec.push_back((l1+l2)/2);
                
                layer6->calc_grads(cases[batch_num].out);
                layer5->calc_grads(layer6->grads_in);
                layer4->calc_grads(layer5->grads_in);
                layer3->calc_grads(layer4->grads_in);
                layer2->calc_grads(layer3->grads_in);
                layer1->calc_grads(layer2->grads_in);
                
                
                float diff_cost = cost_vec[cost_vec.size()-1] - cost_vec[cost_vec.size()-2];
                
                if ( diff_cost < 0.00001 )     // Stops training if cost decreases very slow
                  break;

                layer1->fix_weights(learning_rate);
                layer2->fix_weights(learning_rate);
                layer3->fix_weights(learning_rate);
                layer4->fix_weights(learning_rate);
                layer5->fix_weights(learning_rate);
                layer6->fix_weights(learning_rate);

        }
    }

    cout<<"**********cost*************\n";

    for(int i=0; i<cost_vec.size(); i++){
        cout<<cost_vec[i]<<endl;
    }


	return 0;
}
