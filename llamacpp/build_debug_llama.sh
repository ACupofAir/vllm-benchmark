# source /opt/intel/oneapi/setvars.sh
#source /opt/intel/oneapi/2025.0/oneapi-vars.sh
 
source /home/bmg/intel/oneapi/setvars.sh

cd llama-cpp-bigdl
 
rm -rf build && mkdir build
 
cmake -B build -G "Ninja" -DLLAMA_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx  -DCMAKE_BUILD_TYPE=Debug  -DBUILD_SHARED_LIBS=ON -DGGML_USE_BIGDL=ON -DLLAMA_BUILD_TESTS=OFF
cmake --build build --config Debug -j
 
# cmake -B build -G "Ninja" -DLLAMA_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx  -DCMAKE_BUILD_TYPE=Release  -DBUILD_SHARED_LIBS=ON -DGGML_USE_BIGDL=ON -DLLAMA_BUILD_TESTS=OFF
# cmake --build build --config Release -j
