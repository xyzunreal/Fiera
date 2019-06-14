#include "../../CNN/model.h"
#include "../../CNN/Dataset/MNIST.h"
#include"../../CNN/Dataset.h"

// #include "byteswap.h"



int main()
{
    vector<layer_t* > layers;
    // case_t packet = read_test_cases();

    tdsize batch_size = {-1,28,28,1};
    cout<<"flag\n";
    // conv_layer_t * layer1 = new conv_layer_t(1,2,2,temp_in.size,false,false);
    conv_layer_t * layer1b = new conv_layer_t(1,8,12,batch_size, false, false);
    prelu_layer_t * layer2p = new prelu_layer_t( layer1b->out_size, false, false);
    batch_norm_layer_t * layerbaa = new batch_norm_layer_t(layer2p->out_size);
    conv_layer_bin_t * layer1bb = new conv_layer_bin_t(1,5,15, layerbaa->out_size, false, false);
    prelu_layer_t * layer2pp = new prelu_layer_t( layer1bb->out_size, false, false);
    batch_norm_layer_t * layerbaaa = new batch_norm_layer_t(layer2pp->out_size);
    conv_layer_bin_t * layer1bbb = new conv_layer_bin_t(1,5,20, layerbaaa->out_size, false, false);
    prelu_layer_t * layer2ppp = new prelu_layer_t( layer1bbb->out_size, false, false);
    batch_norm_layer_t * layerbbaaa = new batch_norm_layer_t(layer2ppp->out_size);
    conv_layer_bin_t * layer1bbbb = new conv_layer_bin_t(1,5,30, layerbbaaa->out_size, false, false);
    prelu_layer_t * layer2 = new prelu_layer_t( layer1bbbb->out_size, false, false);
    batch_norm_layer_t * layerba = new batch_norm_layer_t(layer2->out_size);
    fc_layer_bin_t * layerfcb1 = new fc_layer_bin_t(layerba->out_size,{-1, 108, 1, 1});
    fc_layer_t * layer3 = new fc_layer_t(layerfcb1->out_size, {-1, 10, 1, 1});
    // scale_layer_t * layer4 = new scale_layer_t(layer3->out.size);
    softmax_layer_t * layer5 = new softmax_layer_t(layer3->out_size,false);

    // layers.push_back((layer_t *) layer1);
    layers.push_back((layer_t *) layer1b);
    layers.push_back((layer_t *) layer2p);
    layers.push_back((layer_t *) layerbaa);
    layers.push_back((layer_t *) layer1bb);
    layers.push_back((layer_t *) layer2pp);
    layers.push_back((layer_t *) layerbaaa);
    layers.push_back((layer_t *) layer1bbb);
    layers.push_back((layer_t *) layer2ppp);
    layers.push_back((layer_t *) layerbbaaa);
    layers.push_back((layer_t *) layer1bbbb);
    layers.push_back((layer_t *) layer2); 
    layers.push_back((layer_t *) layerba);
    layers.push_back((layer_t *) layerfcb1);
    layers.push_back((layer_t *) layer3);
    // layers.push_back((layer_t *) layer4);
    layers.push_back((layer_t *) layer5);
    // print_tensor_size(packet.out.size);
    Model model(layers);
    // model.summary();
    // model.save_model("batch_32.json");
    model.load_weights("weights_after_3_epochs_lr_1e-5_b32");
    
    Dataset data = load_mnist(0.1,0,0);
    
    // print_tensor_size(data.train.images.size);

    tensor_t<float> output = model.predict(data.train.images);
    
    int crct = 0;

    for(int i=0; i<output.size.m; i++){
        int idx,aidx;
        float maxm = 0.0f;

        for(int j=0; j<10; j++){
            if(output(i,j,0,0)>maxm){
                maxm = output(i,j,0,0);
                idx = j;
            }
            if(int(data.train.labels(i,j,0,0))==1){
                aidx = j;
            }
        }
        if(idx == aidx) crct++;
        cout<<"predicted: "<<idx<<" actual: "<<aidx<<endl;
    }

    cout<<"correct number is "<<crct<<endl;

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
