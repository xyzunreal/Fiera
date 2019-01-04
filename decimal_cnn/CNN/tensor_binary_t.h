#pragma once
#include "point_t.h"
#include <cassert>
#include <vector>
#include <string.h>

//template<typename T>
struct tensor_bit_t
{
	bitset<100000> data;

	tdsize size;

	// bitset can't be initialize dynamically
	//tensor_t( int _x, int _y, int _z )
	//{
		//data = new T[_x * _y * _z];
		//size.x = _x;
		//size.y = _y;
		//size.z = _z;
	//}

	tensor_bit_t(const tensor_bit_t& other )
	{
		//data = new T[other.size.x *other.size.y *other.size.z];
		//memcpy(
			//this->data,
			//ther.data,
			//other.size.x *other.size.y *other.size.z * sizeof( T )
		//);
		this->size = other.size;
		this->data = other.data;
	}

	//~ tensor_t<T> operator+( tensor_t<T>& other )
	//~ {
		//~ tensor_t<T> clone( *this );
		//~ for ( int i = 0; i < other.size.x * other.size.y * other.size.z; i++ )
			//~ clone.data[i] += other.data[i];
		//~ return clone;
	//~ }

	//~ tensor_t<T> operator-( tensor_t<T>& other )
	//~ {
		//~ tensor_t<T> clone( *this );
		//~ for ( int i = 0; i < other.size.x * other.size.y * other.size.z; i++ )
			//~ clone.data[i] -= other.data[i];
		//~ return clone;
	//~ }

	bool operator()( int _x, int _y, int _z )
	{
		
		return this->get( _x, _y, _z );
	}

	bool get( int _x, int _y, int _z )
	{
		assert( _x >= 0 && _y >= 0 && _z >= 0 );
		assert( _x < size.x && _y < size.y && _z < size.z );

		return data[
			_z * (size.x * size.y) +
				_y * (size.x) +
				_x
		];
	}

	void copy_from( std::vector<std::vector<std::vector<T>>> data )
	{
		int z = data.size();
		int y = data[0].size();
		int x = data[0][0].size();

		for ( int i = 0; i < x; i++ )
			for ( int j = 0; j < y; j++ )
				for ( int k = 0; k < z; k++ )
					this->data[k*(size.x*size.y) + j*(size.x) + i] = data[k][j][i];
	}
	
	xnor_builtin_popcount(tensor_bit_t other){
		// 
	}
	
	~tensor_bit_t()
	{
		delete[] data;
	}
};

static void print_tensor( tensor_t data )
{
	int mx = data.size.x;
	int my = data.size.y;
	int mz = data.size.z;

	for ( int z = 0; z < mz; z++ )
	{
		printf( "[Dim%d]\n", z );
		for ( int y = 0; y < my; y++ )
		{
			for ( int x = 0; x < mx; x++ )
			{
				printf( "%.2f \t", (float)data.get( x, y, z ) );
			}
			printf( "\n" );
		}
	}
}

static tensor_t<float> to_tensor( std::vector<std::vector<std::vector<float>>> data )
{
	int z = data.size();
	int y = data[0].size();
	int x = data[0][0].size();


	tensor_t<float> t( x, y, z );

	for ( int i = 0; i < x; i++ )
		for ( int j = 0; j < y; j++ )
			for ( int k = 0; k < z; k++ )
				t( i, j, k ) = data[k][j][i];
	return t;
}