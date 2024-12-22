# Array creation

## 1) 1-D tensor from rust vector

RSTSR tensor can be created by (owned) vector object.

In the following case, memory of vector object `vec` will be transferred to tensor object `tensor`.
Except for relatively small overhead (generating layout of tensor), **no explicit data copy occurs**.

```rust
{{#include ../../listings/fundamentals/listing-01-vec_to_tsr/src/main.rs}}
```

\* Note: This will generate tensor object for default CPU device.
Without further configuration, RSTSR chooses `DeviceFaer` as the default tensor device.
If other devices are of interest (such as single-threaded `DeviceCpuSerial`), or you may wish to confine number of threads for `DeviceFaer`, then you may wish to apply another version of `asarray`.
For example, to limit 4 threads when performing computation, you may initialize tensor by the following code:

```rust
{{#include ../../listings/fundamentals/listing-01-vec_to_tsr_otherdevice/src/main.rs}}
```

## 2) $n$-D tensor from rust vector

For $n$-D tensor, the recommended way to build from existing vector, without explicit memory copy, is
- first, build 1-D tensor from contiguous memory;
- second, reshape to the $n$-D tensor you desire;


