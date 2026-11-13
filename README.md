# transformer
Tiny transformer

## Build

Requires CMake, a C++17 compiler, Eigen3, and GoogleTest (for tests).

```bash
./build.sh
```

The script creates `build/` automatically and builds with 8 parallel jobs.
Use `JOBS=4 ./build.sh` to change the parallel job count. Additional arguments
are passed to CMake, for example `./build.sh -DCMAKE_BUILD_TYPE=Release`.

Run tests with:

```bash
ctest --test-dir build --output-on-failure
```
