#include<bits/stdc++.h>
#include "../../CNN/tensor_t.h"


using namespace std;

int main(){
    tensor_t<float> temp(2,2,2,2);
    temp(1,1,0,0) = 1;  
    // temp.size.x = temp.size.y = temp.size.z = temp.size.m = 2;

    print_tensor(temp);
    
    return 0;
}