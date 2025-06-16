## usage

1. install oneapi-toolkit and source it, using `dpcpp -v` to check if installed
2. compile command example:

- p2p_cpy

```bash
dpcpp p2p_cpy.cpp
./a.out
```

- check_onednn.cpp

```bash
icpx -fsycl check_onednn.cpp -std=c++20 -ldnnl -ltbb -o example
./exmaple
```
