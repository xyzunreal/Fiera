#include "../../CNN/model.h"
#include "../../CNN/Dataset/MNIST.h"
#include"../../CNN/Dataset.h"

using namespace std;
int main()
{
    Model model;

    // PATH to pretrained model
    string PATH="trained_models/big_binary_mnist1";
    #ifdef using_cmake
    PATH="Example\\ MNIST/full_test/trained_models/big_binary_mnist1";
    #endif

    model.load(PATH);   
    model.summary();

    Dataset data = load_mnist(5,5,0); // % of train, test, validation images
    model.train(data.train.images, data.train.labels, 16, 1, 0.0001);

    tensor_t<float> output = model.predict(data.test.images, 1);

     // Calculate no. of images predicted correctly
     int correct = 0;
     for(int i=0; i<output.size.m; i++){
         int idx, aidx;
         float maxm = 0.0f;

         for(int j=0; j<10; j++){
             if(output(i,j,0,0)>maxm){
                 maxm = output(i,j,0,0);
                 idx = j;
             }
             if(int(data.test.labels(i,j,0,0))==1){
                 aidx = j;
             }
         }
         if(idx == aidx) correct++;
     }

     cout<<"correct number is "<<correct<< " / " << output.size.m << endl;

    // If you want to save new weights, uncomment code below

    // model.save(PATH);

    return 0;

}
