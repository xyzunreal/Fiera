
/*! Custom Model class having 
    : train( tensor_t<float>, tensor_t<float>, int, int, float, string, string, bool){
    : predict(tensor_t, bool)
    : tensor_t<float> predict (string)
    : save_model(string)
    : load_model(string)
    : save_weights(string)
    : load_weights(string)
    : load(string)
    : save(string)
    : summary()
*/


#include <cassert>
#include <cstdint>
#include <string.h>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <chrono>  // for high_resolution_clock
#include <sys/stat.h>
#include "byteswap.h"
#include "cnn.h"

using namespace std;
#pragma pack(push, 1)

class Model{

    vector<layer_t* > layers;
    int epochs=0;
    int batch_size;
    int num_of_batches;
    float loss;
    float learning_rate;

    public:
        Model(vector<layer_t *> layers){
            /**
            * Copy layers
            **/
            this->layers = layers;
        }
        Model(){}

        float Step_decay(float epoch){
        /**
        * return 'learning_rate' reduce by a factor of 'drop' for every 'epoch_drop'
        **/

            float drop = 0.5; 
            float epoch_drop = 10;

            float n_learning_rate = learning_rate*pow(drop, floor((1+epoch)/epoch_drop)); 
            this->learning_rate = n_learning_rate;
        }

        void train( tensor_t<float> input, tensor_t<float> output, int batch_size, int epochs=1, float lr = 0.02,
                     string optimizer="Momentum", string lr_schedule = "Step_decay", bool debug=false ){
        //TODO: Check layers are not empty
        
            this->epochs += epochs;
            this->batch_size = batch_size;
            this->num_of_batches = input.size.m / batch_size ;
            this->learning_rate = lr;

            cout<<"Total images: " << input.size.m<<endl;
            cout<<"batch_size: " << batch_size<<endl;
            cout<<"epochs "<<epochs<<endl;

            for ( int epoch = 0; epoch < epochs; ++epoch){
                for(int batch_num = 0; batch_num<num_of_batches; batch_num++)
                {
                    auto start = std::chrono::high_resolution_clock::now();

                    tensor_t<float> input_batch = input.get_batch(batch_size, batch_num);
                    tensor_t<float> output_batch = output.get_batch(batch_size, batch_num);
                    tensor_t<float> out;
                    
                    /*! Forward propogate */
                    for ( int i = 0; i < layers.size(); i++ )
                    {

                        if ( i == 0 )
                            out = activate( layers[i], input_batch, true);
                        else
                            out = activate( layers[i], out, true);
                    }

                    /*! Calculate Loss */
                    this->loss = cross_entropy(out, output_batch, debug);
                    
                    cout <<"loss for epoch: "<< epoch << "/" << epochs << " and batch: " << batch_num << "/" << num_of_batches << " is " << loss << endl;
                    
                    /*!  Backpropogation */
                    tensor_t<float> grads_in;

                    for ( int i = layers.size() - 1; i >= 0; i-- )
                    {

                        if ( i == layers.size() - 1 )
                            grads_in = calc_grads( layers[i], output_batch);
                        else
                            grads_in = calc_grads( layers[i], grads_in );
                     }
                    
                    /*! setup 'drop' and 'epoch_drop' parameters in the function itself*/
                    Step_decay(epoch);

                    /*!  Update weights */
                    for ( int i = 0; i < layers.size(); i++ )
                        fix_weights( layers[i], this->learning_rate);

                    if (!(batch_num % 10)){ 
                        auto finish = std::chrono::high_resolution_clock::now();
                        std::chrono::duration<double> elapsed = finish - start;
                        cout << "Estimated time: " << elapsed.count() / 60.0 * ((num_of_batches - batch_num + 1) + num_of_batches * (epochs - epoch + 1)) <<" minutes" << endl;
                    }
                }
            }
        }

        tensor_t<float> predict(tensor_t<float> input_batch, bool measure_time=false){
            
            /** 
            *given a input_batch returns the forward propagation output 
            **/

            tensor_t<float> out;
            auto start = std::chrono::high_resolution_clock::now();

            for ( int i = 0; i < layers.size(); i++ )
            {
                if ( i == 0 ){
                    out = activate( layers[i], input_batch, false);
                }
                else{
                    out = activate( layers[i], out, false);
                }
            }

            if (measure_time)
            {
                auto finish = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = finish - start;
                std::cout << "Elapsed time: " << elapsed.count() << " s\n";
            }

            return out;
        }
        
        tensor_t<float> predict (string input_image){

            /** 
            *Takes path of input image (only .ppm currently) as input.
            **/
            struct stat buffer;   
            assert(stat (input_image.c_str(), &buffer) == 0);   // Checks if file exist

            ifstream file( input_image, ios::binary | ios::ate ); 
            streamsize size = file.tellg();
            file.seekg( 0, ios::beg );
            assert( size != -1); 
            uint8_t* data = new uint8_t[size];
            file.read( (char*)data, size );
            if ( data )
            {
                uint8_t * usable = data;
                while ( *(uint32_t*)usable != 0x0A353532 )
                    usable++;

    #pragma pack(push, 1)
                struct RGB
                {
                    uint8_t r, g, b;
                };
    #pragma pack(pop)

                RGB * rgb = (RGB*)usable;

                tensor_t<float> image(1, 28, 28, 1);
                tensor_t<float> output;
                for ( int i = 0; i < 28; i++ )
                {
                    for ( int j = 0; j < 28; j++ )
                    {
                        RGB rgb_ij = rgb[i * 28 + j];
                        image( 0, i, j, 0 ) = ((((float)rgb_ij.r
                                    + rgb_ij.g
                                    + rgb_ij.b)
                                    / (3.0f*255.f)));
                    }
				}

                for ( int i = 0; i < layers.size(); i++ )
                {
                    if ( i == 0 )
                        output = activate( layers[i], image, false);
                    else
                        output = activate( layers[i], output, false);
                return output;

                }
			}
        }


        void save_model( string fileName ){
            ofstream file(fileName);
            json model;

            // If any configuration needed later, save here.
            model["config"] = {
                {}
            };
            for ( int i = 0; i < layers.size(); i++ ) 
                save_layer( layers[i], model );
            file << std::setw(4) << model << std::endl;
            cout << "\n Model saved in file " << fileName << endl;
            file.close();
        }

        void load_model( string fileName ){
            
            if (!layers.empty()) {
                cout << "Deleting previous stored layers \n" << endl;
                layers.clear();
            }

            struct stat buffer;
            if (!stat (fileName.c_str(), &buffer) == 0) {
                cout << "File not found: " << fileName << endl;
                exit(0);
            }
            ifstream file(fileName);
            json model;
            file >> model;
            json layersJ = model["layers"];

            for (auto& el : layersJ.items()){
                json layerJ = el.value();
                json inJ = layerJ["in_size"];
                tdsize in_size;
                in_size.from_json(inJ);
                
                if (layerJ["layer_type"] == "fc"){
                    json outJ = layerJ["out_size"];
                    tdsize out_size;
                    out_size.from_json(outJ);
                    fc_layer_t * layer = new fc_layer_t(in_size, out_size );
                    layers.push_back((layer_t *) layer);
                    continue;
                } 

                else if (layerJ["layer_type"] == "conv"){
                    conv_layer_t * layer = new conv_layer_t(layerJ["stride"], layerJ["extend_filter"], layerJ["number_filters"], in_size);
                    layers.push_back((layer_t *) layer);
                    continue;
                }

                else if (layerJ["layer_type"] == "prelu"){
                    prelu_layer_t * layer = new prelu_layer_t(in_size);
                    layer->alpha = layerJ["alpha"];
                    layer->prelu_zero = layerJ["prelu_zero"];
                    layers.push_back((layer_t *) layer);
                    continue;
                }

                else if (layerJ["layer_type"] == "softmax"){
                    softmax_layer_t * layer = new softmax_layer_t(in_size);
                    layers.push_back((layer_t *) layer);
                    continue;
                } 

                else if (layerJ["layer_type"] == "batch_norm2D"){
                    batch_norm_layer_t * layer = new batch_norm_layer_t(in_size);
                    layers.push_back((layer_t *) layer);
                    continue;
                }

                else if (layerJ["layer_type"] == "conv_bin"){
                    conv_layer_bin_t * layer = new conv_layer_bin_t(layerJ["stride"], layerJ["extend_filter"], layerJ["number_filters"], in_size);
                    layers.push_back((layer_t *) layer);
                    continue;
                }

                else if (layerJ["layer_type"] == "fc_bin"){
                    json outJ = layerJ["out_size"];
                    tdsize out_size;
                    out_size.from_json(outJ);
                    fc_layer_bin_t * layer = new fc_layer_bin_t(in_size, out_size);
                    layers.push_back((layer_t *) layer);
                    continue;
                }

                else if (layerJ["layer_type"] == "scale"){
                    scale_layer_t * layer = new scale_layer_t(in_size);
                    layers.push_back((layer_t *) layer);
                    continue;
                }
            }
            cout << "\n Model loaded successfully from " << fileName << endl;
        }

        void save_weights( string folderName ){
            assert(layers.size() > 1);

            mkdir(folderName.c_str(), 0777);
            
            time_t now = time(0);
            string date = ctime(&now);
            json j = {
                {"Date", date},
                {"Epochs", this->epochs},
                {"Num_of_batches", this->num_of_batches},
                {"Batch_size", this->batch_size},
                {"Training_loss", this->loss},
                {"Learning_rate", this->learning_rate}
            };
            ofstream file(folderName + "/metadata.json");
            file << j;
            file.close();

            for ( int i = 0; i < layers.size(); i++ ){
                string fileName = folderName + "/" + to_string(i) + ".weights";
                save_layer_weight( layers[i], fileName );
            }        
            cout << "\n Mode saved in folder " << folderName << endl;
        }

        void load_weights( string folderName ){
            assert(layers.size() > 1);
            ifstream file(folderName+"/metadata.json");
            json j;
            file >> j;
            this->epochs = j["Epochs"];
            this->num_of_batches = j["Num_of_batches"];
            this->batch_size = j["Batch_size"];
            this->loss = j["Training_loss"];
            file.close();
            struct stat buffer;
            for ( int i = 0; i < layers.size(); i++ ){
                string fileName = folderName + "/" + to_string(i) + ".weights";
                if (stat (fileName.c_str(), &buffer) == 0)
                    load_layer_weight( layers[i], fileName);
                else{
                    cout << "File " << fileName << " does not exist\n";
                    exit(0);
                }
            }
            cout << "\n Model loaded from folder " << folderName << endl;
        }

        void load( string folderName ){
            this->load_model( folderName + "/model.json");
            this->load_weights( folderName + "/weights");
        }

        void save( string folderName ){
            mkdir(folderName.c_str(), 0777);
            this->save_model( folderName + "/model.json");
            this->save_weights( folderName + "/weights");
        }

        void summary(){
            cout << "\n\t\t\tMODEL SUMMARY\n";
            for (auto& layer : layers){
                print_layer(layer);
            }
            cout << endl<< endl;
        }
};

#pragma pack(pop)
