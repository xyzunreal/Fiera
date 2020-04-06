#include "../../CNN/model.h"
#include "../../CNN/Dataset/MNIST.h"
#include"../../CNN/Dataset.h"

int main()
{
    vector<layer_t* > layers;

    std::vector<std::vector<std::vector<std::vector<float> > > > vect=
        
        // n c h w

        {{{{-2.9662, -1.0606, -0.3090},
          { 0.9343, -0.3821, -1.1669},
          { 0.3636, -0.3156,  1.1450}},

         {{-0.3822, -0.3553,  0.7542},
          { 0.6901, -0.1443,  1.6120},
          { 1.5671, -1.2432, -1.7178}}},


        {{{-0.5824, -0.6153,  0.4105},
          { 1.7675, -0.0832,  0.5087},
          { 1.1178,  1.1286,  0.1416}},

         {{-0.5458,  1.1542, -1.5366},
          {-0.5577, -0.4383,  1.1572},
          { 0.0889,  0.2659, -0.1907}}}};          

        //n h w c
    tensor_t<float> temp_in(2,3,3,2), predict(2,4,1,1);
    temp_in.from_vector(vect);
    for(int i=0; i<4; i++) 
        for(int img=0;img<2;img++) 
            predict(img,i,0,0) = 0;
    predict(0,3,0,0) = 1;
    predict(1 ,1, 0, 0) = 1;

    tdsize batch_size = {-1,3,3,2};

    conv_layer_t * layer1b = new conv_layer_t(1,2,2,batch_size, false, false);
    // prelu_layer_t * layer2p = new prelu_layer_t( layer1b->out_size, false, false);
    batch_norm_layer_t * layerbaa = new batch_norm_layer_t(layer1b->out_size);

    // conv_layer_bin_t * layer1bb = new conv_layer_bin_t(1,2,2,batch_size, false, false);

    vect = {{{{ 0.1701, -0.1747},
          {-0.1887,  0.3051}},

         {{ 0.3235,  0.0407},
          {-0.0612, -0.0456}}},


        {{{ 0.1675, -0.3301},
          {-0.2889,  0.2824}},

         {{ 0.3490, -0.0210},
          {-0.2794,  0.0097}}}};

    layer1b->filters.from_vector(vect);

    // prelu_layer_t * layer2pp = new prelu_layer_t( layer1bb->out_size, false, false);
    // batch_norm_layer_t * layerba = new batch_norm_layer_t(layer2pp->out_size);

    // conv_layer_bin_t * layer1bbb = new conv_layer_bin_t(1,3,60,layerba->out_size, false, false);
    // prelu_layer_t * layer2ppp = new prelu_layer_t( layer1bbb->out_size, false, false);
    // batch_norm_layer_t * layerbaaa = new batch_norm_layer_t(layer2ppp->out_size);

    fc_layer_t * layer3 = new fc_layer_t(layerbaa->out_size, {-1, 4, 1, 1});
    
    vect = 
        {{{{ 0.1558,  0.0148,  0.0896,  0.170}}},
        {{{-0.3019,  0.0659,  0.0488, -0.1747}}},
        {{{ 0.3323,  0.2685,  0.1723, -0.1887}}},
        {{{-0.2773,  0.0909,  0.3247,  0.3051}}},
        {{{ 0.2707,  0.1876, -0.0787,  0.3235}}},
        {{{-0.0614, -0.2735, -0.1970,  0.0407}}},
        {{{ 0.1819,  0.2517, -0.0890, -0.0612}}},
        {{{ 0.1378,  0.1217, -0.2155, -0.0456}}}};

    layer3->weights.from_vector(vect);

    cout<<"fc weight: "<<endl;
    
    print_tensor(layer3->weights);

    softmax_layer_t * layer5 = new softmax_layer_t(layer3->out_size,false);

    // layers.push_back((layer_t *) layer1b);
    // layers.push_back((layer_t *) layer2p);
    // layers.push_back((layer_t *) layerbaa);
    layers.push_back((layer_t *) layer1b);
    // layers.push_back((layer_t *) layer2pp);
    layers.push_back((layer_t *) layerbaa);
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

    model.train(temp_in, predict, 2, 1, 0.0001);

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
