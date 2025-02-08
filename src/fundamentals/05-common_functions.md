# Common Functions

<div class="warning">

**We will improve further for this part.**

Current RSTSR implements many common functions, and that can cover lots of common usage as a tensor algebra library.
However, there are still many features that has not implemented.

</div>

## 1. Elementwise Functions

For RSTSR, most functions declared in [Python Array API](https://data-apis.org/array-api/2023.12/API_specification/elementwise_functions.html) has been implemented.
Most of them can be called as regular rust function, or as associated methods:

```rust
{{#include ../../listings/features-default/tests/common_functions.rs:example_elem_01}}
```

Note that some exceptions that can only called by associated methods `tensor.method(...)`, and cannot be called as regular rust functions `rt::method(tensor, ...)` currently. These exceptions includes
- unary functions `.abs()`, `.real()`, `.imag()`, `.sign()`;
- binary function `.pow()`.

Consequently, for example `.abs()`, following code is valid:

```rust
{{#include ../../listings/features-default/tests/common_functions.rs:example_elem_02}}
```

However, RSTSR does not have `abs` function.

As a take home message, if not sure how to use elementwise functions, use the associated `tensor.method(...)`.

```rust,ignore,does_not_compile
    rt::abs(&a) // RSTSR does not have `abs` as function

    // This is due to different implementation for real and complex numbers.
    // For real types, both `abs(&a)` and `abs(a)` works; latter one is inplace absolute;
    // - implemented trait TensorRealAbsAPI
    // For complex types, only `abs(&a)` works.
    // - implemented trait TensorComplexAbsAPI
```

## 2. Mapping Functions


