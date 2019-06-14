#include<bits/stdc++.h>
#include "../../CNN/Dataset/MNIST.h"
#include"../../CNN/Dataset.h"

using namespace std;

int main(){
    
    Dataset tdata = load_mnist(60,20,20);
    // cout<<"flag\n";
    print_tensor_size(tdata.train.images.size);
    print_tensor_size(tdata.train.labels.size);
    print_tensor_size(tdata.test.images.size);
    print_tensor_size(tdata.test.labels.size);

    return 0;
}