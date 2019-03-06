#include<bits/stdc++.h>
#include "byteswap.h"
#include "../../CNN/cnn.h"

using namespace std;

int mini_batch_size = 1;

int num_batches = 6000/64;	
	
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

	// cout<<"case_count: "<<case_count<<endl;
	
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

    cout<<"************image**************\n";
    print_tensor(cases[0].data);
    conv_layer_t * layer1 = new conv_layer_t( 1, 8, 1, cases[0].data.size,true);		
    prelu_layer_t * layer2 = new prelu_layer_t( layer1->out.size,true);
    // batch_norm_layer_t * layer2_1 = new batch_norm_layer_t(layer2->out.size);
	// conv_layer_bin_t * layer3 = new conv_layer_bin_t(1, 8, 1, layer2_1->out.size );
	// prelu_layer_t * layer4 = new prelu_layer_t(layer3->out.size);
    // batch_norm_layer_t * layer5 = new batch_norm_layer_t(layer4->out.size);
    fc_layer_t * layer6 = new fc_layer_t(layer2->out.size, 10,true);
    prelu_layer_t * layer7 = new prelu_layer_t(layer6->out.size,true);
    // fc_layer_bin_t * layer8 = new fc_layer_bin_t(layer7->out.size,10);
    // prelu_layer_t * layer9 = new prelu_layer_t(layer8->out.size);
    // scale_layer_t * layer10 = new scale_layer_t(layer7->out.size);
    softmax_layer_t * layer11 = new softmax_layer_t(layer7->out.size,true);

    vector<double> cost_vec;
    cost_vec.push_back(0);
    double learning_rate = 0.01;

   
    bool flag = false;
    for(int epoch = 0; epoch<100; epoch++){

        //batch_num<num_batches
        for(int batch_num = 0; batch_num<1; batch_num++){
                
                layer1->activate(cases[batch_num].data);
                layer2->activate(layer1->out);
                // layer2_1->activate(layer2->out); 
                // layer3->activate(layer2_1->out);
                // layer4->activate(layer3->out);
                // layer5->activate(layer4->out);
                layer6->activate(layer2->out);
                layer7->activate(layer6->out);
                // layer8->activate(layer7->out);
                // layer9->activate(layer8->out);
                // layer10->activate(layer7->out);
                layer11->activate(layer7->out);

                
                tensor_t<double> costs = cross_entropy(layer11->out, cases[batch_num].out, false);
                double costs_avg = 0;

                for(int e = 0; e<mini_batch_size; e++){
                    costs_avg += costs(e,0,0,0);
                }

                costs_avg /= (double)mini_batch_size;
            
                if(isnan(costs_avg)){
                    flag = true;
                    cout<<"*****************exit**************\n";
                    break;
                }

                cost_vec.push_back(costs_avg);
                
                int size = 11;

                // cross_entropy(layers[size-1]->out, expected);
                
                int tm = layer11->out.size.m;
                int tx = layer11->out.size.x;

                tensor_t<double> softmax_grads(tm,tx,1,1);

                for(int i=0; i<tm; i++){
                    for(int j=0; j<tx; j++){
                        if(int(cases[batch_num].out(i,j,0,0)) == 1){
                            softmax_grads(i,j,0,0) = (-1.0/layer11->out(i,j,0,0));
                        }
                    }
                }

                layer11->calc_grads(softmax_grads);
                // layer10->calc_grads(layer11->grads_in);
                // layer9->calc_grads(layer10->grads_in); 
                // layer8->calc_grads(layer9->grads_in);
                layer7->calc_grads(layer11->grads_in);
                layer6->calc_grads(layer7->grads_in);
                // layer5->calc_grads(layer6->grads_in);
                // layer4->calc_grads(layer5->grads_in);
                // layer3->calc_grads(layer4->grads_in);
                // layer2_1->calc_grads(layer3->grads_in);
                layer2->calc_grads(layer6->grads_in);
                layer1->calc_grads(layer2->grads_in);
                
                
                double diff_cost = cost_vec[cost_vec.size()-1] - cost_vec[cost_vec.size()-2];
                
                // if(diff_cost < 0.00001){
                //     learning_rate -= (learning_rate/20.0);
                // }

                layer1->fix_weights(learning_rate);
                layer2->fix_weights(learning_rate);
                // layer2_1->fix_weights(learning_rate);
                // layer3->fix_weights(learning_rate);
                // layer4->fix_weights(learning_rate);
                // layer5->fix_weights(learning_rate);
                layer6->fix_weights(learning_rate);
                layer7->fix_weights(learning_rate);
                // layer8->fix_weights(learning_rate);
                // layer9->fix_weights(learning_rate);
                // layer10->fix_weights(learning_rate);
                layer11->fix_weights(learning_rate);

        }
        if(flag){
            break;
        }
    }

    cout<<"**********cost*************\n";

    for(int i=0; i<cost_vec.size(); i++){
        cout<<cost_vec[i]<<endl;
    }

    cout<<"learning_rate\n";
    cout<<learning_rate<<endl;

	return 0;
}
