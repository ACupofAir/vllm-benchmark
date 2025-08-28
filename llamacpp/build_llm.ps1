$llm_cpp_dir = "C:\Users\ADC\workspace\llm.cpp\bigdl-core-xe\llama_backend"

Push-Location

Set-Location $llm_cpp_dir

if (Test-Path "build") {
    Remove-Item -Path "build" -Recurse -Force
}


cmake -B build -G "Ninja" -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx -DCMAKE_BUILD_TYPE=Release
cmake --build build -j -v

Move-Item build/llama_bigdl_core.lib build/libllama_bigdl_core.lib

Pop-Location
