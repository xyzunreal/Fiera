#include<bits/stdc++.h>
#include "tensor_bin_t.h"
#include "tensor_t.h"
#include "conv_layer_t.h"

using namespace std;

int main(){

    // vector<layer_t*> layers;
	
	tensor_t<double> temp_in(1, 5,5,1);
	
	for(int i=0; i<5; i++){
		for(int j=0; j<5; j++){
			int maxval = 25;
			temp_in(0, i,j,0) = 2.19722f / maxval * (rand()-rand()) / double( RAND_MAX );
		}
	}
	
	
	for(int i=0; i<5; i++){
		for(int j=0; j<5; j++){
			temp_in(0, i,j,0) = pow(-1,i^j)*2+i+j-4;
		}
	}
	
	//debug
	cout<<"*********input image *******"<<endl;
	
	for(int x=0; x<5; x++){
		for(int y=0; y<5; y++){
			//debug
			cout<<temp_in(0, x,y,0)<<' ';
		}
		//debug
		cout<<endl;
	}
    return 0;
}