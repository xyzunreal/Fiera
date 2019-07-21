
/*! Custom tensor*/

// REMEMBER: tensor_bin_t(m, x, y, z) returns the position in linear array
//           while tensor_t(m, x, y, z) returns the reference

#pragma once
#include "point_t.h"
#include "../Libraries/json.hpp"
#include "gradient_t.h"
#include <cassert>
#include <vector>
#include <cmath>
#include <string.h>
#include <fstream>
#include <inttypes.h> //!< For printing uint64_t

using namespace std;
using json = nlohmann::json;

bool areEqual(float a,float b) {
	
	/** 
     * @param a, b float variables
	 * @return true if a equals b upto two decimal points
     */

	return trunc(100. * a) == trunc(100. * b);
}

template<typename T>
struct tensor_t
{
	/** 
     * Custom 4D tensor holding two variables i.e. data and size(m,x,y,z)
     */

	T * data;
	// static int ccount, dcount;
	tdsize size;

	tensor_t(){
	}
	
	tensor_t(int _m, int _x, int _y, int _z)
	{
		/** 
     	* @param _m, _x, _y, _z 
		* @return Constructor to declare tensor having size of _m, _x, _y, _z
		*/

		// ccount+=1;
		data = new T[_x * _y * _z * _m];
		memset(data,0,sizeof(T)*_x*_y*_z*_m);
		size.m = _m;
		size.x = _x;
		size.y = _y;
		size.z = _z;
	}

	tensor_t(tdsize sz)
	{
		/** 
     	* @param tdsize sz 
		* @return Constructor to declare tensor having size of 'sz'
		*/
		// ccount++;
		data = new T[sz.x * sz.y * sz.z * sz.m];
		memset(data,0,sizeof(T)*sz.x*sz.y*sz.z*sz.m);
		size.x = sz.x;
		size.y = sz.y;
		size.m = sz.m;
		size.z = sz.z;
	}

	tensor_t( const tensor_t& other )
	{
		/** 
     	* @param tensor_t other 
		* @return cloned tensor with 'other' tensor values   
		*/

		// ccount++;
		data = new T[other.size.x *other.size.y *other.size.z * other.size.m];
		memcpy(
			this->data,
			other.data,
			other.size.x *other.size.y *other.size.z * other.size.m * sizeof( T )
		);
		// this->data = other.data;
		this->size = other.size;
	}

	tensor_t<T> operator+( tensor_t<T>& other )
	{
		/** 
		* overloaded '+' operator to add two tensors 
     	* @param tensor_t other, this
		* @return summation of other and this   
		*/

		assert(this->size == other->size);

		tensor_t<T> clone( *this );
		for ( int i = 0; i < other.size.x * other.size.y * other.size.z * other.size.m; i++ )
			clone.data[i] += other.data[i];
		return clone;
	}

	tensor_t<T> operator-( tensor_t<T>& other )
	{
		/** 
		* overloaded '-' operator to add two tensors 
     	* @param tensor_t other, this 
		* @return difference of other and this   
		*/
	
		tensor_t<T> clone( *this );
		for ( int i = 0; i < other.size.x * other.size.y * other.size.z* other.size.m; i++ )
			clone.data[i] -= other.data[i];
		return clone;
	}

	bool operator==( tensor_t<T> t2 )
	{
		/** 
		* overloaded '==' operator to add two tensors 
     	* @param tensor_t t2, this 
		* @return whether t2 equals this   
		*/

		tensor_t<T> t1( *this );	
		bool equal = false;
		for ( int i = 0; i < t1.size.x * t1.size.y * t1.size.z* t1.size.m; i++ )
			if ( !areEqual( t1.data[i], t2.data[i] ) )	return false;   // Cannot use '==' because of precision loss
		return true;	
	}

	T& operator()( int _m, int _x, int _y, int _z)
	{
		/** 
		* to access the elements at [_m][_x][_y][_z] position in tensor 
     	* @param integer _m, _x, _y, _z 
		* @return integer at the given position   
		*/
	
		return this->get(_m, _x, _y, _z);
	}

	void operator = (tensor_t<T> t){
		// ccount++;

		/** 
		* transfer the value of tensor t to this tensor  
     	* @param integer _m, _x, _y, _z 
		* @return integer at the given position   
		*/

		if(t.size.m != size.m or t.size.x != size.x or t.size.y != size.y or t.size.z != size.z){

			data = new T[t.size.m * t.size.x * t.size.y * t.size.z];
			memset(data,0,sizeof(T)*t.size.m * t.size.x * t.size.y * t.size.z);
			size.m = t.size.m ;
			size.x = t.size.x ;
			size.y = t.size.y;
			size.z = t.size.z;
		}
		for(int i=0; i<t.size.x*t.size.y*t.size.m*t.size.z; i++)
			this->data[i] = t.data[i];
	}

	void resize(tdsize sz){
		
		this->size = sz;
		data = new T[size.m * size.x * size.y * size.z];
		memset(data,0,sizeof(T)*size.m * size.x * size.y * size.z);	
	}	

	T& get(int _m, int _x, int _y, int _z)
	{
		return data[
			_m * (size.x * size.y * size.z) +
				_z * (size.x * size.y) +
				_y * size.x +
				 _x ];
	}

	void from_vector( std::vector<std::vector<std::vector<std::vector<T> > > > data )
	{
		// data is saved as [m][z][y][x]

		int m = data.size();
		int z = data[0].size();
		int y = data[0][0].size();
		int x = data[0][0][0].size();


		for( int tm = 0; tm<m; tm++ )
			for ( int i = 0; i < x; i++ )
				for ( int j = 0; j < y; j++ )
					for ( int k = 0; k < z; k++ )
						get( tm, i, j, k) = data[tm][k][j][i];

	}

	tensor_t<float> get_batch(int batch_size, int batch_num){
		
		/** 
		* return a batch of 'batch_size' starting from 'batch_num'     
		*/

		tensor_t<float> t(batch_size , this->size.x, this->size.y, this->size.z );	
		int start = batch_num * batch_size;
		int end = ( batch_num + 1 ) * batch_size;

		for(int tm = start; tm < end; tm++)
			for ( int i = 0; i < this->size.x; i++ )
				for ( int j = 0; j < this->size.y; j++ )
					for ( int k = 0; k < this->size.z; k++ )
						t( tm - start, i, j, k) = this->get(tm, i, j, k);
		return t;
	}

	~tensor_t()
	{
		// dcount++;
		// free(data);
		delete[] data;
	}
};

template<typename T>
void print_tensor( tensor_t<T>& data )
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
					printf( "%.4f \t", (float)data(tm, x, y, z));
				printf( "\n" );
			}
		}
	}
}

template<typename T>
void print_tensor_t( tensor_t<T>& data )
{
	/** 
	* Print tensor having uint64_t data values   
	*/

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
					printf( "%" PRId64 "\t",data(tm, x, y, z));
				printf( "\n" );
			}
		}
	}
}

void print_tensor(tensor_t<gradient_t>& data){

	/** 
	* Printing tensort of gradient_t type values   
	*/

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
					printf( "%.4f \t", (float)data(tm, x, y, z).grad);
				printf( "\n" );
			}
		}
	}	
}

void print_tensor_size(tdsize data){

	cout<<data.m<<",  "<<data.x<<",  "<<data.y<<", "<<data.z<<endl;

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
