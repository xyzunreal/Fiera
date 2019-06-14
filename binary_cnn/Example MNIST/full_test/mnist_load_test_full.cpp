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
  
    // print_tensor_size(packet.out.size);
    Model model;
    // model.train(packet.data, packet.out, 16, 1000, 0.01);
    // model.summary();
    // model.save_model("layers.json");
    // // Model model;
    model.load_model("layers.json");
    
    // model.load_weights("weights");
    // model.save_weights("weights");
    // model.train(temp_in, predict, 2, 10);
    // model.save_weights("weights_after_training");
    // model.load_weights("weights_after_training");
    // model.train(temp_in, predict, 2, 10);
    return 0;
}
