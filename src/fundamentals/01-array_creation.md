# Array creation

## 1. Converting Rust Vector to RSTSR Tensor

### 1.1 1-D tensor from rust vector

RSTSR tensor can be created by (owned) vector object.

In the following case, memory of vector object `vec` will be transferred to tensor object `tensor`[^1].
Except for relatively small overhead (generating layout of tensor), **no explicit data copy occurs**.

```rust
{{#include ../../listings/features-default/tests/array_creation.rs:example_01}}
```

[^1]: This will generate tensor object for default CPU device.
Without further configuration, RSTSR chooses `DeviceFaer` as the default tensor device, with all threads visible to rayon.
If other devices are of interest (such as single-threaded `DeviceCpuSerial`), or you may wish to confine number of threads for `DeviceFaer`, then you may wish to apply another version of `asarray`.
For example, to limit 4 threads when performing computation, you may initialize tensor by the following code:

```rust
{{#include ../../listings/features-default/tests/array_creation.rs:example_02}}
```

### 1.2 $n$-D tensor from rust vector

For $n$-D tensor, the recommended way to build from existing vector, without explicit memory copy, is
- first, build 1-D tensor from contiguous memory;
- second, reshape to the $n$-D tensor you desire;

```rust
{{#include ../../listings/features-default/tests/array_creation.rs:example_03}}
```

We do not recommend generating $n$-D tensor from nested vectors, i.e. `Vec<Vec<T>>`.
Explicit memory copy will always occur anyway in this case.
So for nested vectors, you may wish to first generate a flattened `Vec<T>`, then perform reshape on this:
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

[^2]: Initialization of `TensorView` by rust slices `&[T]` is performed by `ManuallyDrop` internally.
For the data types `T` that scientific computation concerns (such as `f64`, `Complex<f64>`), it will not cause memory leak.
However, if type `T` has its own deconstructor (`drop` function), you may wish to double check for memory leak safety.

## 3. Intrinsic RSTSR Tensor Creation Functions

### 3.1 1-D tensor creation functions

Most useful 1-D tensor creation functions are `arange` and `linspace`.

`arange` creates arrays with regularly incrementing values.
Following code shows multiple ways to generate arrays[^3].

```rust
{{#include ../../listings/features-default/tests/array_creation.rs:example_arange}}
```

[^3]: Many RSTSR functions, especially array creation functions, are signature-overloaded.
Input should be wrapped by tuple to pass multiple function parameters.

`linspace` will create arrays with a specified number of elements, and spaced equally between the specified beginning and end values.

```rust
{{#include ../../listings/features-default/tests/array_creation.rs:example_linspace}}
```

### 3.2 2-D tensor creation functions

Most useful 2-D tensor creation functions is `eye` and `diag`.

`eye` generates identity matrix.
In many cases, you may just provide the number of rows, and `eye(n_row)` will return a square identity matrix, or `eye((n_row, &device))` if device is of concern.
If you may wish to generate a rectangular identity matrix with offset, you may call `eye((n_row, n_col, offset))`.

```rust
{{#include ../../listings/features-default/tests/array_creation.rs:example_eye}}
```

`diag` generates diagonal 2-D tensor from 1-D tensor, or generate 1-D tensor from diagonal of 2-D tensor.
`diag` is defined as overloaded function; if offset of diagonal is of concern, you may wish to call `diag((&tensor, offset))`.

```rust
{{#include ../../listings/features-default/tests/array_creation.rs:example_diag}}
```
