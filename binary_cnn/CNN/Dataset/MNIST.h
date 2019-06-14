
#include "../Dataset.h"
// #include "../byteswap.h"
#include <fstream>

// #include "../model.h"
uint8_t* read_file( const char* szFile )
{
	ifstream file( szFile, ios::binary | ios::ate );
    streamsize size = file.tellg();

	file.seekg( 0, ios::beg );

	if ( size == -1 )
		return nullptr;

	uint8_t* buffer = new uint8_t[size];
	file.read( (char*)buffer, size);
	// cout<<"buffer: "<<buffer[0]<<endl;
    return buffer;
}

Dataset read_test_cases(float ptrain, float ptest, float pval)
{
	uint8_t* train_image = read_file( "../../CNN/Dataset/MNIST/train-images.idx3-ubyte" );
	uint8_t* train_labels = read_file( "../../CNN/Dataset/MNIST/train-labels.idx1-ubyte" );
	
    // uint32_t case_count = 60000;
    // cout<<(*(uint8_t*)(train_image));
	uint32_t case_count = byteswap_uint32( *(uint32_t*)(train_image + 4) );
    
    // cout<<case_count<<endl;

	// calculating number of each division
    int ntrain = floor(case_count*(ptrain/100.f)),
        ntest = floor(case_count*(ptest/100.f)),
        nval = floor(case_count*(pval/100.f));

    // print_tensor_size({ntrain,ntest,nval}); 

    assert(ntrain+ntest+nval <= case_count);

    int crnt_count = 0;

    Dataset fdata(ntrain, ntest, nval, 10, 28, 28, 1);
    
    for( int i = 0; i < case_count; i++ )
	{

		uint8_t* img = train_image + 16 + i * (28 * 28);

		uint8_t* label = train_labels + 8 + i;

        if(i < ntrain)
        {
            for ( int x = 0; x < 28; x++ )
                for ( int y = 0; y < 28; y++ ){
                    fdata.train.images(crnt_count, x, y, 0) = img[x + y * 28] / 255.f;
                }

            for ( int b = 0; b < 10; b++ )
                fdata.train.labels(crnt_count , b, 0, 0 ) = *label == b ? 1.0f : 0.0f;

            crnt_count++;

            if(crnt_count == ntrain) crnt_count = 0;
            continue;
        }


        if(i < ntrain + ntest){
		// cout<<"flagcc "<<i<<' '<<(ntrain+ntest)<<endl;
            for ( int x = 0; x < 28; x++ )
                for ( int y = 0; y < 28; y++ ){
                    fdata.test.images(crnt_count, x, y, 0) = img[x + y * 28] / 255.f;
                }

            for ( int b = 0; b < 10; b++ )
                fdata.test.labels(crnt_count , b, 0, 0 ) = *label == b ? 1.0f : 0.0f;

            crnt_count++;
            if(crnt_count == ntest) crnt_count = 0;
            continue;
        }
        if(i < ntrain + ntest + nval){
            for ( int x = 0; x < 28; x++ )
                for ( int y = 0; y < 28; y++ ){
                    fdata.validation.images(crnt_count, x, y, 0) = img[x + y * 28] / 255.f;
                }

            for ( int b = 0; b < 10; b++ )
                fdata.validation.labels(crnt_count , b, 0, 0 ) = *label == b ? 1.0f : 0.0f;

            crnt_count++;
        }
        
	}

	delete[] train_image;
	delete[] train_labels;

	return fdata;
}

Dataset load_mnist(float percnt_train, float percnt_test, float percnt_val){

    return read_test_cases(percnt_train, percnt_test, percnt_val);
}