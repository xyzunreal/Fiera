#include<bits/stdc++.h>
//#include"point.h"
#include"conv_layer.h"

using namespace std;

int main(){
	tdsize in_size;
	in_size.x = in_size.y = in_size.z = 3;
	
	conv_layer aa(1, 2, 10, in_size);
	print_tensor(aa.filters[0]);
	return 0;
}
