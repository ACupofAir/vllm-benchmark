source /opt/intel/oneapi/2025.0/oneapi-vars.sh --force

cd ~/junwang/llm.cpp/bigdl-core-xe/llama_backend

rm -rf build && mkdir build

cmake -B build -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release
cmake --build build -j -v

sudo cp build/libllama_bigdl_core.a /usr/lib
sudo ls /usr/lib -l | grep libllama # check libllama_bigdl_core.a
