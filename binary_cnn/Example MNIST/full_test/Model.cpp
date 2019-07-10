#include "../../CNN/model.h"
#include "../../CNN/Dataset/MNIST.h"
#include"../../CNN/Dataset.h"

int main()
{
    vector<layer_t* > layers;
    std::vector<std::vector<std::vector<std::vector<float> > > > vect=
          

        
    tensor_t<float> temp_in(2,3,3,2), predict(2,4,1,1);
    temp_in.from_vector(vect);
    for(int i=0; i<4; i++) 
        for(int img=0;img<2;img++) 
            predict(img,i,0,0) = 0;
    predict(0,3,0,0) = 1;
    predict(1 ,1, 0, 0) = 1;

    tdsize batch_size = {-1,4,4,2};

    // conv_layer_t * layer1b = new conv_layer_t(1,3,14,batch_size, false, false);
    // prelu_layer_t * layer2p = new prelu_layer_t( layer1b->out_size, false, false);
    // batch_norm_layer_t * layerbaa = new batch_norm_layer_t(layer2p->out_size);

    conv_layer_bin_t * layer1bb = new conv_layer_bin_t(1,2,2,batch_size, false, false);

    vect = {{{{ 0.1701, -0.1747},
          {-0.1887,  0.3051}},

         {{ 0.3235,  0.0407},
          {-0.0612, -0.0456}}},


        {{{ 0.1675, -0.3301},
          {-0.2889,  0.2824}},

         {{ 0.3490, -0.0210},
          {-0.2794,  0.0097}}}};

    layer1bb->filters.from_vector(vect);

    // prelu_layer_t * layer2pp = new prelu_layer_t( layer1bb->out_size, false, false);
    // batch_norm_layer_t * layerba = new batch_norm_layer_t(layer2pp->out_size);

    // conv_layer_bin_t * layer1bbb = new conv_layer_bin_t(1,3,60,layerba->out_size, false, false);
    // prelu_layer_t * layer2ppp = new prelu_layer_t( layer1bbb->out_size, false, false);
    // batch_norm_layer_t * layerbaaa = new batch_norm_layer_t(layer2ppp->out_size);

    fc_layer_t * layer3 = new fc_layer_t(layer1bb->out_size, {-1, 4, 1, 1});
    softmax_layer_t * layer5 = new softmax_layer_t(layer3->out_size,false);

    // layers.push_back((layer_t *) layer1b);
    // layers.push_back((layer_t *) layer2p);
    // layers.push_back((layer_t *) layerbaa);
    layers.push_back((layer_t *) layer1bb);
    // layers.push_back((layer_t *) layer2pp);
    // layers.push_back((layer_t *) layerba);
    // layers.push_back((layer_t *) layer1bbb);
    // layers.push_back((layer_t *) layer2ppp);
    // layers.push_back((layer_t *) layerbaaa);
    layers.push_back((layer_t *) layer3);
    layers.push_back((layer_t *) layer5);
    
    // tensor_t<float>::ccount = 0;
    // tensor_t<float>::dcount = 0;
    Model model(layers);
    model.summary();

    string PATH="trained_models/big_binary_mnist";

    #ifdef using_cmake
    PATH="Example\\ MNIST/full_test/trained_models/big_binary_mnist";
    #endif

    // Dataset data = load_mnist(60,10,0);
    // model.load("PATH");

    model.train(temp_in, predict, 1, 20, 0.0001);

    // model.train(data.train.images, data.train.labels, 56, 2, 0.0002);
    

    // model.save(PATH);
    // model.save(PATH);


    // tensor_t<float> output = model.predict(data.test.images, 1);
    // int correct = 0;

    // for(int i=0; i<output.size.m; i++){
    //     int idx,aidx;
    //     float maxm = 0.0f;

    //     for(int j=0; j<10; j++){
    //         if(output(i,j,0,0)>maxm){
    //             maxm = output(i,j,0,0);
    //             idx = j;
    //         }
    //         if(int(data.test.labels(i,j,0,0))==1){
    //             aidx = j;
    //         }
    //     }
    //     if(idx == aidx) correct++;
    //     cout<<"predicted: "<<idx<<" actual: "<<aidx<<endl;
    // }

    // cout<<"correct number is "<<correct<< " / " << output.size.m << endl;

    // cout<<"******Constructor called********* "<<tensor_t<float>::ccount<<endl;
    // cout<<"*****Destructor called*********** "<<tensor_t<float>::dcount<<endl;

    // model.train(packet.data, packet.out, 16, 2, 0.001);
    // model.save_weights("weights_after_1_epoch_lr_1e-5_b32");
    // model.train(packet.data, packet.out, 16, 2, 0.00001);
    // model.save_weights("weights_after_3_epochs_lr_1e-5_b32");
    // mod el.train(packet.data, packet.out, 16, 100, 0.00001);
    // model.save_weights("weights_after_13_epochs_lr_1e-4_b32");
    // model.train(packet.data, packet.out, 16, 10, 0.00001);
    // model.save_weights("weights_after_23_epochs_lr_1e-5_b32");
    // model.train(packet.data, packet.out, 16, 100, 0.00001);
    // model.save_weights("weights_after_unlimited_fully_on_lr_1e-5_b32");
    return 0;

}
