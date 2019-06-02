#include <cassert>
#include <cstdint>
#include <string.h>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <sys/stat.h>
#include "byteswap.h"
#include "cnn.h"

using namespace std;
#pragma pack(push, 1)
class Model{

    vector<layer_t* > layers;
    public:
        Model(vector<layer_t *> layers){
            this->layers = layers;
    }
        Model(){}

        void train( tensor_t<float> input, tensor_t<float> output, int batch_size, int epochs=1, float lr = 0.02, string optimizer="Momentum", bool debug=false ){
        //TODO: Check layers are not empty

            int num_of_batches = input.size.m / batch_size ;
            float loss;

            
            for ( int epoch = 0; epoch < epochs; ++epoch){
                for(int batch_num = 0; batch_num<num_of_batches; batch_num++)
                {
                    tensor_t<float> input_batch = input.get_batch(batch_size, batch_num);
                    tensor_t<float> output_batch = output.get_batch(batch_size, batch_num);
                    
                    // Forward propogate
                    for ( int i = 0; i < layers.size(); i++ )
                    {
                        if ( i == 0 )
                            activate( layers[i], input_batch );
                        else
                            activate( layers[i], layers[i - 1]->out );
                    }

                    // Calculate Loss
                    loss = cross_entropy(layers[layers.size()-1]->out, output_batch, debug);

                    // Backpropogation
                    for ( int i = layers.size() - 1; i >= 0; i-- )
                    {
                        if ( i == layers.size() - 1 )
                            calc_grads( layers[i], output_batch);
                        else
                            calc_grads( layers[i], layers[i + 1]->grads_in );
                     }

                    // Update weights
                    for ( int i = 0; i < layers.size(); i++ )
                        fix_weights( layers[i], lr);
                }
                cout<<"Loss after epoch "<<epoch<<" : "<< loss <<endl;
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
                    fc_layer_t * layer = new fc_layer_t(in_size, layerJ["out_size"]);
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
                } 
            }
            cout << "\n Model loaded successfully from " << fileName << endl;
        }

        void save_weights( string folderName ){
            for ( int i = 0; i < layers.size(); i++ ){
                mkdir(folderName.c_str(), 0777);
                string fileName = folderName + "/" + to_string(i) + ".weights";
                save_layer_weight( layers[i], fileName );
            }
            cout << "\n Mode saved in folder " << folderName << endl;
        }

        void load_weights( string folderName ){
            assert(layers.size() > 1);
            for ( int i = 0; i < layers.size(); i++ ){
                string fileName = folderName + "/" + to_string(i) + ".weights";
                load_layer_weight( layers[i], fileName);
            }
            cout << "\n Model loaded from folder " << folderName << endl;
        }

        void summary(){
            cout << "\n\t\t\tMODEL SUMMARY\n";
            for (auto& layer : layers)
                print_layer(layer);
            cout << endl<< endl;
        }
};

#pragma pack(pop)
