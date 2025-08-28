$llama_cpp_dir = "C:\Users\ADC\workspace\llama-cpp-bigdl"

Push-Location

Set-Location $llama_cpp_dir

if (Test-Path "build") {
    Remove-Item -Path "build" -Recurse -Force
}

cmake -G "Ninja" -B build `
    -DGGML_SYCL=ON `
    -DCMAKE_C_COMPILER=icx `
    -DCMAKE_CXX_COMPILER=icx `
    -DBUILD_SHARED_LIBS=ON `
    -DCMAKE_BUILD_TYPE=Release `
    -DGGML_USE_BIGDL=ON `
    -DLLAMA_BUILD_TESTS=OFF `
    -DGGML_NATIVE=OFF `
    -DGGML_AVX=ON `
    -DGGML_AVX2=ON `
    -DLLAMA_CURL=OFF

cmake --build build -j

Pop-Location
