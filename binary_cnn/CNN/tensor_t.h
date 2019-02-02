#pragma once
#include "point_t.h"
#include <cassert>
#include <vector>
#include <string.h>

template<typename T>
struct tensor_t
{
	T * data;

	tdsize size;

	
				/*to be deleted*/
	tensor_t(int _x, int _y, int _z){
		data = new T[_x * _y * _z];
		memset(data,0,sizeof(T)*_x*_y*_z);
		size.x = _x;
		size.y = _y;
		size.z = _z;

	}
				/*to be deleted*/


	tensor_t(int _m, int _x, int _y, int _z)
	{
		data = new T[_x * _y * _z * _m];
		memset(data,0,sizeof(T)*_x*_y*_z*_m);
		size.m = _m;
		size.x = _x;
		size.y = _y;
		size.z = _z;
	}

	tensor_t( const tensor_t& other )
	{
		data = new T[other.size.x *other.size.y *other.size.z * other.size.m];
		memcpy(
			this->data,
			other.data,
			other.size.x *other.size.y *other.size.z * other.size.m * sizeof( T )
		);
		this->size = other.size;
	}

	tensor_t<T> operator+( tensor_t<T>& other )
	{
		tensor_t<T> clone( *this );
		for ( int i = 0; i < other.size.x * other.size.y * other.size.z * other.size.m; i++ )
			clone.data[i] += other.data[i];
		return clone;
	}

	tensor_t<T> operator-( tensor_t<T>& other )
	{
		tensor_t<T> clone( *this );
		for ( int i = 0; i < other.size.x * other.size.y * other.size.z* other.size.m; i++ )
			clone.data[i] -= other.data[i];
		return clone;
	}

	bool operator==( tensor_t<T> t2 )
	{
		tensor_t<T> t1( *this );	
		bool equal = false;
		for ( int i = 0; i < t1.size.x * t1.size.y * t1.size.z* t1.size.m; i++ )
			if (t1.data[i] != t2.data[i])	return false;
		return true;	
	}

					/*to be deleted*/
	T& operator()(int _x, int _y, int _z)
	{
		return this->get(0, _x, _y, _z);
	}
					/*to be deleted*/


	T& operator()( int _m, int _x, int _y, int _z)
	{
		return this->get(_m, _x, _y, _z);
	}

	T& get(int _m, int _x, int _y, int _z)
	{
		assert( _m >=0 &&_x >= 0 && _y >= 0 && _z >= 0 );
		assert( _m < size.m && _x < size.x && _y < size.y && _z < size.z );

		return data[
			_m * (size.x * size.y * size.z) +
				_z * (size.x * size.y) +
				_y * size.x +
				 _x ];
	}

	void from_vector( std::vector<std::vector<std::vector<std::vector<T> > > > data )
	{
		// data is [m][z][y][x]

		int m = data.size();
		int z = data[0].size();
		int y = data[0][0].size();
		int x = data[0][0][0].size();

		for(int tm = 0; tm<m; tm++)
			for ( int i = 0; i < x; i++ )
				for ( int j = 0; j < y; j++ )
					for ( int k = 0; k < z; k++ )
						get( tm, i, j, k) = data[tm][k][j][i];
	}

	~tensor_t()
	{
		delete[] data;
	}
};

void print_tensor( tensor_t<float>& data )
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
				{
					// indexing changed
					printf( "%.6f \t", (float)data(tm, x, y, z));
				}
				printf( "\n" );
			}
		}
	}
}

static tensor_t<float> to_tensor( std::vector<std::vector<std::vector<std::vector<float> > > > data )
{
	int m = data.size();
	int z = data[0].size();
	int y = data[0][0].size();
	int x = data[0][0][0].size();

	tensor_t<float> t(m, x, y, z );

	for(int tm = 0; tm<m; tm++)
		for ( int i = 0; i < x; i++ )
			for ( int j = 0; j < y; j++ )
				for ( int k = 0; k < z; k++ )
					t( tm, i, j, k) = data[tm][k][j][i];
	return t;
}
