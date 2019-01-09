#include<bits/stdc++.h>
#include "../../CNN/tensor_bin_t.h"


using namespace std;

int main(){
    tensor_bin_t temp(2,2,2,2);
    temp.data[temp(1,1,0,0)] = 1;  
    // temp.size.x = temp.size.y = temp.size.z = temp.size.m = 2;

    print_tensor_bin(temp);
    
    return 0;
}