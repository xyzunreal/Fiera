
/*! Custom binarized tensor */

// REMEMBER: tensor_bin_t(m, x, y, z) returns the position in linear array
//           while tensor_t(m, x, y, z) returns the reference

//TODO: Adding debug flags to ifdef
//	  : Upgrade bitset declaration from compile time to runtime 

#pragma once
#include "point_t.h"
#include <bitset>

using namespace std;

struct tensor_bin_t
{
	
	bitset<10000000> data; // Currently we used Static bitset of size 1e5 
	tdsize size;
	tensor_bin_t(int _m, int _x, int _y, int _z )
	{
		data = 0;
		size.m = _m;
		size.x = _x;
		size.y = _y;
		size.z = _z;
	}
	tensor_bin_t(const tensor_bin_t& other )
	{
		this->size = other.size;
		this->data = other.data;
	}
	tensor_bin_t(){
		
	}

	void resize(tdsize sz){
		this->size = sz;
		data = 0;
	}	

	int operator()(int _m, int _x, int _y, int _z)
	{
		return this->get(_m, _x, _y, _z);
	}

	void operator = (tensor_bin_t & other){
		this->size = other.size;
		this->data = other.data;
	}

	int get( int _m, int _x, int _y, int _z)
	{
		// data is accessed as ( m, x, y, z)
		assert( _m >=0 &&_x >= 0 && _y >= 0 && _z >= 0 );
		assert( _m < size.m && _x < size.x && _y < size.y && _z < size.z );

		// return the position in linear 
		return (
			_m * (size.x * size.y * size.z) +
				_z * (size.x * size.y) +
				_y * size.x +
				 _x );
	}

	void copy_from( vector < vector < vector < vector < float> > > > data )
	{
		// data is saved as [m][z][y][x]

		int m = data.size();
		int z = data[0].size();
		int y = data[0][0].size();
		int x = data[0][0][0].size();

		for( int tm = 0; tm < m; tm++)
			for ( int i = 0; i < x; i++ )
				for ( int j = 0; j < y; j++ )
					for ( int k = 0; k < z; k++ )
						this->data[tm*(size.x*size.y*size.z) + k*(size.x*size.y) + j*(size.x) + i] = data[m][k][j][i];
	}
	
	
	~tensor_bin_t()
	{
		//~ delete[] data;
	}
};

static void print_tensor_bin( tensor_bin_t data )
{
	int mx = data.size.x;
	int my = data.size.y;
	int mz = data.size.z;
	int mm = data.size.m;

	for(int tm = 0; tm < mm; tm++){
		
		printf("[Example %d]\n", tm);

		for ( int z = 0; z < mz; z++ )
		{
			printf( "[Dim%d]\n", z );
			for ( int y = 0; y < my; y++ )
			{
				for ( int x = 0; x < mx; x++ )
					cout<<data.data[data( tm, x, y, z)]<<' ';
				printf( "\n" );
			}
		}
	}
}