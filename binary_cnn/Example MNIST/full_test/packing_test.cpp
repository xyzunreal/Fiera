#include <bits/stdc++.h>
#include "../../Libraries/json.hpp"
using namespace std;
using json = nlohmann::json;

template <size_t bitsetsize>
vector<uint8_t> pack(bitset<bitsetsize> input){

    uint8_t OutByte = 0;
    int shiftCounter = 0;
    vector<uint8_t> output;
    // auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < input.size(); i++){
        if (input[i])
            OutByte |= (1 << shiftCounter);
        shiftCounter ++;
        if (shiftCounter > 7)
        {
            output.push_back(OutByte);
            OutByte = 0;
            shiftCounter = 0;
        }
    }
    return output;
    // auto finish = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed = finish - start;
    // std::cout << "Elapsed time: " << elapsed.count() << " s\n";
}

template <size_t bitsetsize>
vector<bool> unpack( vector<uint8_t> input ){

    int shiftCounter = 0;
    vector<bool> output;
    // auto start = std::chrono::high_resolution_clock::now();
    for ( int i = 0; i < input.size(); i++ )
        for (uint8_t shiftCounter = 0; shiftCounter < 8; shiftCounter++)
            output.push_back(input[i] & (1<<shiftCounter));
    return output;
    // auto finish = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed = finish - start;
    // std::cout << "Elapsed time: " << elapsed.count() << " s\n";
}
int main(){
    bitset<10> input;
    vector<uint8_t> a = pack(input);
    json j_pack;
    j_pack["pack_weights"] = a;
    vector<uint8_t> v_pack = json::to_cbor(j_pack);
    ofstream file("pack_weights_cbor.json");
    for (const auto &e : v_pack) file<< e;
}