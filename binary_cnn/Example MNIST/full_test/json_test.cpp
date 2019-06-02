#include <iostream>
#include "../../Libraries/json.hpp"
#include "../../CNN/model.h"
#include <fstream>
#include <bitset>
#include <iomanip>
#include <chrono>  // for high_resolution_clock
using namespace std;
using json = nlohmann::json;
int main() {
json j;
float arr[20] = {0};

json j2 = {
  {"pi", 3.141},
  {"happy", true},
  {"name", "Niels"},
  {"nothing", nullptr},
  {"answer", {
    {"everything", 42}
  }},
  {"list", {1, 0, 2}},
  {"object", {
    {"currency", "USD"},
    {"value", 42.99}
  }},
  {"arr", arr}
};
j2["arr"] = arr;
cout<<j2;

json weights = {
  {"type", "conv"},
  {"size", {1,2,3,4}},
  {"data", {4,55,34,4}},
};
cout << weights;

// {
//   auto start = std::chrono::high_resolution_clock::now();
//   json j_bin;
//   bool arr[1000000] = {0};
//   j_bin["bin_weights"] = arr;
//   ofstream file("bin_weights.json");
//   file << j_bin << std::endl;
//   auto finish = std::chrono::high_resolution_clock::now();
//   std::chrono::duration<double> elapsed = finish - start;
//   std::cout << "Elapsed time: " << elapsed.count() << " s\n";
// }
// {
//   auto start = std::chrono::high_resolution_clock::now();
//   json j_float;
//   float arr_float[1000000] = {1.453486};
//   j_float["float_weights"] = arr_float;
//   ofstream file("float_weights.json");
//   file << std::setw(4) << j_float << std::endl;
//   auto finish = std::chrono::high_resolution_clock::now();
//   std::chrono::duration<double> elapsed = finish - start;
//   std::cout << "Elapsed time: " << elapsed.count() << " s\n";
// }
// {
//   auto start = std::chrono::high_resolution_clock::now();
//   json j_float;
//   float arr_float[1000000] = {1.453486};
//   j_float["float_weights"] = arr_float;
//   ofstream file("float_weights_without_setw.json");
//   file << j_float << std::endl;
//   auto finish = std::chrono::high_resolution_clock::now();
//   std::chrono::duration<double> elapsed = finish - start;
//   std::cout << "Elapsed time: " << elapsed.count() << " s\n";
// }
// {
//   auto start = std::chrono::high_resolution_clock::now();
//   json j_bin;
//   bool arr[1000000] = {0};
//   j_bin["bin_weights"] = arr;
//   std::vector<std::uint8_t> v_bson = json::to_bson(j_bin);
//   ofstream file("bin_weights_bson.json");
//   for (const auto &e : v_bson) file<< e;
//   auto finish = std::chrono::high_resolution_clock::now();
//   std::chrono::duration<double> elapsed = finish - start;
//   std::cout << "Elapsed time: " << elapsed.count() << " s\n";
// }
// {
//   auto start = std::chrono::high_resolution_clock::now();
//   json j_float;
//   float arr[1000000] = {100000.544343};
//   j_float["float_weights"] = arr;
//   vector<uint8_t> v_bson = json::to_bson(j_float);
//   ofstream file("float_weights_bson.json");
//   for (const auto &e : v_bson) file<< e;
//   auto finish = chrono::high_resolution_clock::now();
//   chrono::duration<double> elapsed = finish - start;
//   std::cout << "Elapsed time: " << elapsed.count() << " s\n";
// }
// {
//   auto start = std::chrono::high_resolution_clock::now();
//   json j_bin;
//   bool arr[1000000];
//   j_bin["bin_weights"] = arr;
//   std::vector<std::uint8_t> v_bson = json::to_cbor(j_bin);
//   ofstream file("bin_weights_cbor.json");
//   for (const auto &e : v_bson) file<< e;
//   auto finish = std::chrono::high_resolution_clock::now();
//   std::chrono::duration<double> elapsed = finish - start;
//   std::cout << "Elapsed time: " << elapsed.count() << " s\n";
// }
// {
//   auto start = std::chrono::high_resolution_clock::now();
//   json j_bin;
//   float arr[1000000] = {87.878787};
//   j_bin["bin_weights"] = arr;
//   std::vector<std::uint8_t> v_bson = json::to_cbor(j_bin);
//   ofstream file("float_weights_cbor.json");
//   for (const auto &e : v_bson) file<< e;
//   auto finish = std::chrono::high_resolution_clock::now();
//   std::chrono::duration<double> elapsed = finish - start;
//   std::cout << "Elapsed time: " << elapsed.count() << " s\n";
// }
// {
//   auto start = std::chrono::high_resolution_clock::now();
//   json j_bin;
//   uint8_t arr[1000000] = {87};
//   j_bin["bin_weights"] = arr;
//   std::vector<std::uint8_t> v_bson = json::to_cbor(j_bin);
//   ofstream file("int_weights_cbor.json");
//   for (const auto &e : v_bson) file<< e;
//   auto finish = std::chrono::high_resolution_clock::now();
//   std::chrono::duration<double> elapsed = finish - start;
//   std::cout << "Elapsed time: " << elapsed.count() << " s\n";
// }
// {
//   auto start = std::chrono::high_resolution_clock::now();
//   bool arr[1000000] = {0};
//   ofstream file("bool_manual_weights_cbor.json");
//   for (const bool &e : arr) file<< e;
//   auto finish = std::chrono::high_resolution_clock::now();
//   std::chrono::duration<double> elapsed = finish - start;
//   std::cout << "Elapsed time: " << elapsed.count() << " s\n";
// }
// {
//   auto start = std::chrono::high_resolution_clock::now();
//   bitset<1000000> arr;
//   ofstream file("bitset_manual_weights_cbor.json");
//   file << arr;
//   auto finish = chrono::high_resolution_clock::now();
//   chrono::duration<double> elapsed = finish - start;
//   std::cout << "Elapsed time: " << elapsed.count() << " s\n";
// }

return 0;
}
