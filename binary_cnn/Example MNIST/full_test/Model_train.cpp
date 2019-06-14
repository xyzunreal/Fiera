#include "../../CNN/model.h"
// #include "byteswap.h"

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

case_t read_test_cases()
{
    
	uint8_t* train_image = read_file( "../train-images.idx3-ubyte" );
	uint8_t* train_labels = read_file( "../train-labels.idx1-ubyte" );
	
	// cout<<(train_image[0])<<endl;
	
	uint32_t case_count = byteswap_uint32( *(uint32_t*)(train_image + 4) );

	// cout<<"case_count: "<<case_count<<endl;
	
    int crnt_count = 0;

	case_t c {tensor_t<float>( case_count, 28, 28, 1 ), tensor_t<float>(case_count, 10, 1, 1 )};
 
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

	}

	delete[] train_image;
	delete[] train_labels;

	return c;
}



int main()
{
    vector<layer_t* > layers;
    case_t packet = read_test_cases();

    tdsize batch_size = {16,28,28,1};
    // conv_layer_t * layer1 = new conv_layer_t(1,2,2,temp_in.size,false,false);
    conv_layer_t * layer1b = new conv_layer_t(1,5,20,batch_size, false, false);
    prelu_layer_t * layer2p = new prelu_layer_t( layer1b->out_size, false, false);
    batch_norm_layer_t * layerbaa = new batch_norm_layer_t(layer2p->out_size);
    conv_layer_t * layer1bb = new conv_layer_t(1,5,50, layerbaa->out_size, false, false);
    prelu_layer_t * layer2pp = new prelu_layer_t( layer1bb->out_size, false, false);
    batch_norm_layer_t * layerbaaa = new batch_norm_layer_t(layer2pp->out_size);
    // conv_layer_t * layer1bbb = new conv_layer_t(1,3,8, layerbaaa->out_size, false, false);
    // prelu_layer_t * layer2ppp = new prelu_layer_t( layer1bbb->out_size, false, false);
    // batch_norm_layer_t * layerbbaaa = new batch_norm_layer_t(layer2ppp->out_size);
    // conv_layer_t * layer1bbbb = new conv_layer_t(1,3,12, layerbbaaa->out_size, false, false);
    // prelu_layer_t * layer2 = new prelu_layer_t( layer1bbbb->out_size, false, false);
    // batch_norm_layer_t * layerba = new batch_norm_layer_t(layer2->out_size);
    fc_layer_t * layer3 = new fc_layer_t(layerbaaa->out_size, {-1,500,1,1});
    fc_layer_t * layer4 = new fc_layer_t(layer3->out_size, {-1,10,1,1});

    // scale_layer_t * layer4 = new scale_layer_t(layer3->out_size);
    softmax_layer_t * layer5 = new softmax_layer_t(layer4->out_size,false);

    // layers.push_back((layer_t *) layer1);
    layers.push_back((layer_t *) layer1b);
    layers.push_back((layer_t *) layer2p);
    layers.push_back((layer_t *) layerbaa);
    layers.push_back((layer_t *) layer1bb);
    layers.push_back((layer_t *) layer2pp);
    layers.push_back((layer_t *) layerbaaa);
    // layers.push_back((layer_t *) layer1bbb);
    // layers.push_back((layer_t *) layer2ppp);
    // layers.push_back((layer_t *) layerbbaaa);
    // layers.push_back((layer_t *) layer1bbbb);
    // layers.push_back((layer_t *) layer2); 
    // layers.push_back((layer_t *) layerba);
    layers.push_back((layer_t *) layer3);
    layers.push_back((layer_t *) layer4);
    layers.push_back((layer_t *) layer5);

    // print_tensor_size(packet.out.size);
    Model model(layers);
    model.train(packet.data, packet.out, 16, 1000, 0.01);
    // model.summary();
    // model.save_model("layers.json");
    // // Model model;
    // model.load_model("layers.json");
    // model.save_weights("weights");
    // model.load_weights("weights");
    // model.train(temp_in, predict, 2, 10);
    // model.save_weights("weights_after_training");
    // model.load_weights("weights_after_training");
    // model.train(temp_in, predict, 2, 10);
    return 0;
}
