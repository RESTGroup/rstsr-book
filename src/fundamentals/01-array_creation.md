# Array creation

## 1. Converting Rust Vector to RSTSR Tensor

### a. 1-D tensor from rust vector

RSTSR tensor can be created by (owned) vector object.

In the following case, memory of vector object `vec` will be transferred to tensor object `tensor`[^1].
Except for relatively small overhead (generating layout of tensor), **no explicit data copy occurs**.

```rust
{{#include ../../listings/features-default/tests/array_creation.rs:example_01}}
```

[^1]: Note: This will generate tensor object for default CPU device.
Without further configuration, RSTSR chooses `DeviceFaer` as the default tensor device, with all threads visible to rayon.
If other devices are of interest (such as single-threaded `DeviceCpuSerial`), or you may wish to confine number of threads for `DeviceFaer`, then you may wish to apply another version of `asarray`.
For example, to limit 4 threads when performing computation, you may initialize tensor by the following code:

```rust
{{#include ../../listings/features-default/tests/array_creation.rs:example_02}}
```

### b. $n$-D tensor from rust vector

For $n$-D tensor, the recommended way to build from existing vector, without explicit memory copy, is
- first, build 1-D tensor from contiguous memory;
- second, reshape to the $n$-D tensor you desire;

```rust
{{#include ../../listings/features-default/tests/array_creation.rs:example_03}}
```

### c. 2-D tensor from nested rust vector

A special case for generating 2-D tensor is using nested rust vector.
This can be performed by following code, but please note that explicit memory copy is applied, which is not efficient.

```rust
{{#include ../../listings/features-default/tests/array_creation.rs:example_04}}
```

## 2. Converting Rust Slices to RSTSR 1-D TensorView

Rust language is extremely sensitive to ownership of variables, unlike python.
For rust, reference of contiguous memory of data is usually represented as slice `&[T]`.
For RSTSR, this is stored by `TensorView`[^2].

```rust
{{#include ../../listings/features-default/tests/array_creation.rs:example_05}}
```

[^2]: Note: Initialization of `TensorView` by rust slices `&[T]` is performed by `ManuallyDrop` internally.
For the data types `T` that scientific computation concerns, it will not cause memory leak.
However, if type `T` has its own deconstructor (`drop` function), you may wish to double check for memory leak safety.
