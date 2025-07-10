ONEAPI=2025.0
source /opt/intel/oneapi/$ONEAPI/oneapi-vars.sh --force
 

cd ~/junwang/llama-cpp-bigdl
rm -rf build-$ONEAPI && mkdir build-$ONEAPI
 
#cmake -B build-$ONEAPI -G "Ninja" \
#cmake -B build-$ONEAPI \
#    -DGGML_SYCL=ON \
#    -DCMAKE_C_COMPILER=icx \
#    -DCMAKE_CXX_COMPILER=icpx \
#    -DCMAKE_BUILD_TYPE=Release \
#    -DBUILD_SHARED_LIBS=ON \
#    -DGGML_USE_BIGDL=ON \
#    -DLLAMA_BUILD_TESTS=OFF

#DEBUG
cmake -B build-$ONEAPI \
    -DGGML_SYCL=ON \
    -DCMAKE_C_COMPILER=icx \
    -DCMAKE_CXX_COMPILER=icpx \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS="-U_GLIBCXX_ASSERTIONS" \
    -DGGML_USE_BIGDL=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_AVX=ON \
    -DGGML_AVX2=ON \
    -DLLAMA_CURL=OFF

#RUONAN
#cmake -B build-$ONEAPI \
#    -DGGML_SYCL=ON \
#    -DCMAKE_C_COMPILER=icx \
#    -DCMAKE_CXX_COMPILER=icpx \
#    -DGGML_USE_BIGDL=ON \
#    -DLLAMA_BUILD_TESTS=OFF \
#    -DGGML_NATIVE=OFF \
#    -DGGML_AVX=ON \
#    -DGGML_AVX2=ON \
#    -DLLAMA_CURL=OFF

cmake --build build-$ONEAPI --config Release -j


