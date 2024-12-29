# Basic Slicing and Indexing

It is a very common usage to extract sub-matrix from matrix, or indexing tensor to lower dimension sub-tensor, to perform future computation.

RSTSR provides most functionality that NumPy calls "basic indexing", which gives tensor view instead of owned tensor.
By this mechanism, most tensor extraction operation can be performed without memory copy.
For large tensors, **cost of all basic slicing and indexing operations are cheap**, compared to memory assignment and tensor arithmetics.

Due to language limit, in rust, indexing by brackets `[]` can only return underlying data `&T`, so it is not able to return a tensor view by brackets `[]` technically.
In RSTSR, elementwise indexing by `[]` will return reference of element `&T`, only if data is stored by `Vec<T>` type. Usage of `[]` indexing is quite limited. 

However, to obtain sub-tensor view `TensorView` (or `TensorMut`) by using function to index and slice is possible.
The most important functions and macros to perform slicing are
- `slice` (equivalntly `i`): return tensor view by feeding slice parameters;
- `slice_mut` (equivalntly `i`): return mutable tensor view by feeding slice parameters;
- `slice!((start, ) stop (, slice))`: generate slice configuration, which should be similar to python's intrinsic `slice` function.
- `s![]`: generate slice parameters (useful when different types of slicing and indexing occurs in the same time); in most scenarios this macro can be substituted by tuple of different types;

<div class="warning">

**Macro `slice!` is different to function `slice`**.

If you are not feeling good using both function `slice` and macro `slice!` (such as `tensor.slice(slice!(1, 5, 2))`), you can still use the equilvant function `i` to perform tensor indexing and slicing (such as `tensor.i(slice!(1, 5, 2))`).

Clashed naming of these functions may be terrible, but it actually binds to some conventions:
- function `slice` comes from rust crate `ndarray`;
- function `i` comes from rust crate `candle`;
- macro `slice!` comes from python's intrinsic function.

</div>

Note that we have not implemented advanced indexing.
Advanced indexing is mainly about indexing by integer tensor, by boolean tensor, or by index list.
These are well covered in numpy, but will be difficult for RSTSR.
In most cases, advanced indexing requires (or more efficient when there is) explicitly memory copy.
We will persuit to realize some of advanced indexing features in future.

<div class="warning">

**Slicing in RSTSR always generate dynamic dimension.**

Please note that by slicing, RSTSR will always generate dynamic dimension (`IxD`) tensor, instead of generating fixed dimension (`Ix1` for 1-D, `Ix2` for 2-D, etc.).
This is a fallback compared to `ndarray`, where `ndarray` have a more sophisticated macro system to handle fixed dimension slicing.

To convert fixed and dynamic dimension, you may wish to use `into_dim::<D>()` function.
Fixed dimension will be more efficient than dynamic dimension; however, in many arithmetic computations, depneding on contiguous of tensors, the efficiency difference will not be very notable.

</div>

## Terminology

- **Slicing** (by range or slice): $n$-D tensor to $n$-D tensor operation, giving a view of smaller tensor;
- **Indexing** (by integer): $n$-D tensor to $(n-1)$-D tensor, margining out one dimension by selecting;
- **Elementwise Indexing** (by list of integer): give reference of element `&T` instead of giving tensor view.

In RSTSR, slicing and indexing are implemented in a similar way. User can usually simutanously perform slicing and indexing, whenever rust allows.

RSTSR follows rust, C and python convention of 0-based indexing, which is different to Fortran.

## 1. Indexing by Number

For example, a 3-D tensor $A_{ijk}$ can be indexed into 2-D tensor $B_{jk} = A_{2jk}$:

```rust
{{#include ../../listings/features-default/tests/indexing.rs:example_index_by_num_01}}
```

Further more, if you wish to perform indexing to both $i = 2, j = 0$, or say $C_k = A_{20k}$, then you can pass `[2, 0]` into `slice` function:

```rust
{{#include ../../listings/features-default/tests/indexing.rs:example_index_by_num_02}}
```

RSTSR also accepts negative indices for indexing from the end of the array:

```rust
{{#include ../../listings/features-default/tests/indexing.rs:example_index_by_num_03}}
```

## 2. Basic Slicing

### 2.1 Slicing by range

For example, we want to extract $1 \leq i < 3$ from tensor $A_{ijk}$:

```rust
{{#include ../../listings/features-default/tests/indexing.rs:example_index_by_range_01}}
```

First two dimensions slicing are also available by the following way:

```rust
{{#include ../../listings/features-default/tests/indexing.rs:example_index_by_range_02}}
```

Negative indices are also applicable for this case:

```rust
{{#include ../../listings/features-default/tests/indexing.rs:example_index_by_range_03}}
```

### 2.2 Slicing by ranges

Not only range types (like `1..3`) is accepted in RSTSR, but also range to (`..3`) or range from (`1..`).

```rust
{{#include ../../listings/features-default/tests/indexing.rs:example_index_by_range_04}}
```

But as a remainder, rust does not allow two different types to be merged as rust array `[T]`:

```rust,ignore,does_not_compile
    // generate 3-D tensor A_ijk
    let a = rt::arange(24).into_shape([4, 3, 2]).into_owned();

    // different types can't be merged into rust array
    // - `..` is RangeFull
    // - `1..3` is Range
    // - `..2` is RangeTo
    let b = a.slice([.., 1..3, ..2]);
```

To resolve this problem, you may use `s!` macro, or either just pass tuple `(T1, T2)` instead of rust array `[T]`:

```rust
{{#include ../../listings/features-default/tests/indexing.rs:example_index_by_range_05}}
```

We just implemented tuple up to 10 elements; if your tensor is extremely high in number of dimensions, you may wish to use `s!`.

## 3. Special Indexing

### 3.1 Slicing with strides

To slice with stride, you may use `slice!` macro.
The usage of `slice!` macro is similar to python's intrinsic function `slice`[^1]:
- `slice!(stop)`: similar to range to `..stop`;
- `slice!(start, stop)`: similar to range `start..stop`;
- `slice!(start, stop, step)`: this is similar to fortran's or numpy's slicing `start:stop:step`.

[^1]: In `ndarray`, this is done by `s![start..stop;step]`. `ndarray`'s resolution is more concise. However, we stick to use the seemingly verbose `slice!` macro to generate strided slice.

```rust
{{#include ../../listings/features-default/tests/indexing.rs:example_slice_with_strides_01}}
```

In many cases, `None` is also valid input for `slice!`. In fact, `slice!` is realized by mechanics of `Option<T>`, so using `Some(val)` is also valid.

```rust
{{#include ../../listings/features-default/tests/indexing.rs:example_slice_with_strides_02}}
```

### 3.2 Inserting axes

You can insert axes by `None` or `NewAxis` (by definition `Indexer::Insert`). This is similar to numpy's `None` or `np.newaxis`.

```rust
{{#include ../../listings/features-default/tests/indexing.rs:example_insert_axes_01}}
```

Using `None` can be elegent, however, we do not accept `Some(val)` for indexing. So although the following code compiles, it simply does not work.


```rust,panics
{{#include ../../listings/features-default/tests/indexing.rs:example_insert_axes_02}}
```

### 3.3 Ellipsis

In RSTSR, you may use `Ellipsis` (by definition `Indexer::Ellipsis`) to skip some indexes:

```rust
{{#include ../../listings/features-default/tests/indexing.rs:example_ellipsis_01}}
```

### 3.4 Mixed indexing and slicing

As mentioned before, using array type `[T]` is not suitable for representing various kinds of indexing and slicing.
However, you may use macro `s!` or tuple to perform this task[^2].

[^2]: In most cases, macro `s!` and tuple works in the same way; however, they have different definitions in program. `s!` should work in more scenarios.

```rust
{{#include ../../listings/features-default/tests/indexing.rs:example_mixed_indexing}}
```

## 4. Elementwise Indexing

<div class="warning">

**Elementwise indexing is not efficient.**

We also offer elementwise indexing in RSTSR.
But please note that, in most cases, elementwise indexing is not efficient.
- for "unchecked" elementwise indexing, it have more chance to prevent compiler's internal vectorize and SIMD optimization;
- for "safe" elementwise indexing, additional out-of-bound check is performed, further hampering optimizations.

Thus, for computationally intensive tasks, you are encouraged to use RSTSR internal arithmetic functions or mapping functions, to avoid direct elementwise indexing.
Only use elementwise indexing when efficiency is not of concern, or RSTSR internal functions could not fulfill your demands.

</div>

### 4.1 Safe elementwise indexing

To perform indexing, you may use rust's bracket `[]`:

```rust
{{#include ../../listings/features-default/tests/indexing.rs:example_elementwise_safe}}
```

If you provides index out-of-bound, RSTSR will panic:

```rust,panics
{{#include ../../listings/features-default/tests/indexing.rs:example_elementwise_safe_panic}}
```

It is different in RSTSR in indexing (to tensor view) and elementwise indexing (to reference of value).

```rust
{{#include ../../listings/features-default/tests/indexing.rs:example_elementwise_by_tensor_index}}
```

### 4.2 Unchecked elementwise indexing

Unchecked elemtwise indexing will be slightly faster than safe elementwise indexing.
To perform indexing, you may use unsafe function `index_uncheck`:

```rust
{{#include ../../listings/features-default/tests/indexing.rs:example_elementwise_unchecked}}
```

If you provides index out-of-bound, if the index is still smaller than the underlying memory size, RSTSR will not panic and give wrong value:

```rust,not_desired_behavior
{{#include ../../listings/features-default/tests/indexing.rs:example_elementwise_unchecked_not_desired}}
```

This function is marked `unsafe` in order to avoid such kind of out-of-bound (but not out-of-memory).
In most cases it is still memory safe, in that out-of-memory accessing `Vec<T>` will gracefully panics.
