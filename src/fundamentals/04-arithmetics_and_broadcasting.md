# Arithmetics and Broadcasting

As a tensor toolkit, many basic arithmetics are available in RSTSR.

We will touch arithmetics only in this section, and will mention computations based on mapping in next section.

## 1. Examples of Arithmetics Operations

RSTSR can handle `+`, `-`, `*`, `/` operations:

```rust
{{#include ../../listings/features-default/tests/arithmetics_and_broadcasting.rs:basic_arithmetics_01}}
```

RSTSR can handle matmul operations by operator `%` (matrix-matrix, matrix-vector or vector-vector inner dot, and has been optimized in some devices such as `DeviceFaer`):

```rust
{{#include ../../listings/features-default/tests/arithmetics_and_broadcasting.rs:basic_arithmetics_02}}
```

For some special cases, bit operations and shift are also available:

```rust
{{#include ../../listings/features-default/tests/arithmetics_and_broadcasting.rs:basic_arithmetics_03}}
```

The aforementioned examples should have coverd most usages of tensor arithmetics.
The following document in this section will cover some advanced topics.

## 2. Overrided Operator `%`

We have already shown that `%` is the operator for matrix multiplication. This is RSTSR specific usage.
This may cause some confusion, and we will discuss this topic.

Firstly, we follow convention of numpy that `*` will always be elementwise multiply, similar to `+`.

```rust
{{#include ../../listings/features-default/tests/arithmetics_and_broadcasting.rs:star_as_elem_mult}}
```

Numpy introduces `@` notation for matrix multiplication by version 1.10 with [PEP 465](https://peps.python.org/pep-0465/).
For rust, it is virtually hopeless to use the same `@` operator as matrix multiplication, which is fully discussed in [Rust internal forum](https://internals.rust-lang.org/t/add-operator-for-matrix-multiplication/16026/17) (`@` has been used as binary operator for [pattern binding](https://doc.rust-lang.org/book/appendix-02-operators.html)).
To the RSTSR developer's perspective, this is very unfortunate.

Also, other kind of operators (such as `%*%` for R, `.*` for Matlab and Julia, `.` for Mathematica) simply don't exist as binary operator in rust's language.
If we wish to that kind of notations, it requires support from programming language level, and this kind of features are not promised to be stablized soon.

However, we consider that though `%` has been commonly used as remainder, it is less used in vector or matrix computation.
`%` also shares the same operator priority with `*` and `/`.
Thus, we decided to apply `%` as matrix multiplication notation if proper.

As a result, remainder function can not be easily called by RSTSR. Currently, a temporary workaround is using `rstsr_core::tensor::operators::rem`:
```rust
{{#include ../../listings/features-default/tests/arithmetics_and_broadcasting.rs:true_rem}}
```

## 3. Broadcasting

[Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) makes many tensor operations very simple.
RSTSR applies most broadcasting rules from numpy or [Python Array API](https://data-apis.org/array-api/2023.12/API_specification/broadcasting.html).
We refer interested users to numpy and Python Array API documents.

RSTSR initial developer is a computational chemist. We will use an example in chemistry programming, to show how to use broadcasting in real-world situations.

### 3.1 Example of elementwise multiplication

Sum-of-exponent approximation to RI-MP2 (resolution-identity Moller-Plesset second order perturbation), also termed as LT-OS-MP2, involves the following computation:
$$
\mathcal{Y}_{Pia} = Y_{Pia} \epsilon_{i} \epsilon_{a}
$$

```rust
{{#include ../../listings/features-default/tests/arithmetics_and_broadcasting.rs:lt_os_mp2_01}}
```

This is elementwise multiplication of 3-D tensor with 1-D tensors. In usual cases, the 1-D tensors $\epsilon_{i}$ and $\epsilon_{a}$ should be expanded and repeated to 3-D counterpart $E^\mathrm{occ}_{Pia} = \epsilon_i (\forall P, a)$ and $E^\mathrm{vir}_{Pia} = \epsilon_a (\forall P, i)$, then perform multiplication
$$
\mathcal{Y}_{Pia} = Y_{Pia} E^\mathrm{occ}_{Pia} E^\mathrm{vir}_{Pia}
$$
This is both inconvenient and inefficient. By broadcasting, we can insert axis to 1-D tensors, without repeating values:
$$
\mathcal{Y}_{Pia} = Y_{Pia} \epsilon_{\cdot i \cdot} \epsilon_{\cdot \cdot a}
$$

```rust
{{#include ../../listings/features-default/tests/arithmetics_and_broadcasting.rs:lt_os_mp2_02}}
```

This multiplication can still be simplified.
By numpy's definition of broadcasting rule, it will always add ellipsis at the first dimension.
So any operation that inserts axis at the first dimension can be removed:
$$
\mathcal{Y}_{Pia} = Y_{Pia} \epsilon_{i \cdot} \epsilon_{a}
$$

```rust
{{#include ../../listings/features-default/tests/arithmetics_and_broadcasting.rs:lt_os_mp2_03}}
```

Finally, for memory and efficiency concern, it is preferred to perform tensor elementwise multiplication of $\epsilon_{i \cdot} \epsilon_{a}$ first:
$$
\mathcal{Y}_{Pia} = Y_{Pia} (\epsilon_{i \cdot} \epsilon_{a})
$$

```rust
{{#include ../../listings/features-default/tests/arithmetics_and_broadcasting.rs:lt_os_mp2_04}}
```

### 3.2 Example of matrix multiplication

Many post-HF methods involve integral basis transformation, mostly from raw basis (atomic basis or denoted AO, for example) to molecular orbital basis (denoted MO):
$$
Y_{P ai} = \sum_{\mu \nu} Y_{P \mu \nu} C_{\mu i} C_{\nu a}
$$
This operation involves five indices, $P, \mu, \nu, a, i$, where number of indices $a, i$ are smaller than $\mu, \nu$.

```rust
{{#include ../../listings/features-default/tests/arithmetics_and_broadcasting.rs:ao2mo_vo_01}}
```

The [broadcasting rule](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.matmul.html) is slightly complicated for matrix multiplication.
However, if you are familiar to broadcasting rule, this task can be realized with very simple code:

```rust
{{#include ../../listings/features-default/tests/arithmetics_and_broadcasting.rs:ao2mo_vo_02}}
```

<div class="warning">

**This operation can be further optimized in efficiency.**

This code is simple and elegant. It will properly handle multi-threading in devices with rayon support.

However, it requires multiple times of accessing 3-D tensors, and will generate a temporary 3-D tensor.
This is both inefficient in memory access and memory cost.

To resolve memory inefficiency problem, this computation can be performed with parallel axis iterator.
However, RSTSR has not finished this part currently.
We will touch this topic in a later time.

</div>

## 4. Memory Aspects

This is related to how value is passed to arithmetic operations.

In rust, variable ownership and lifetime rule is strict. The following code will give compiler error:

```rust,does_not_compile
    let a = rt::arange(5.0);
    let b = rt::arange(5.0) + 1.0;

    let c = a + b;
    let d = a * b;
```

```console
    |     let c = a + b;
    |                 - value moved here
    |     let d = a * b;
    |                 ^ value used here after move
    |
help: consider cloning the value if the performance cost is acceptable
    |
    |     let c = a + b.clone();
    |                  ++++++++
```

However, in many cases, performance and memory cost of cloning the tensor is not acceptable.
So it is more preferred to perform computation by the following ways, to avoid memory copy and lifetime limitations:
- use reference of tensor,
- use view of tensor,
- use clone of view of tensor,

```rust
{{#include ../../listings/features-default/tests/arithmetics_and_broadcasting.rs:memory_aspects_01}}
```

It should be noted that, except for lifetime limitation, owned tensor is still able to be passed to arithmetic operations.
Moreover, inplace arithmetics will be applied when possible (type constraint and broadcastability).
For example of 1-D tensor addition, memory of variable `c` is not allocated, but instead reused from variable `a`.
So if you are sure that `a` will not be used anymore, you can pass `a` by value, and that will be more efficient.

```rust
{{#include ../../listings/features-default/tests/arithmetics_and_broadcasting.rs:memory_aspects_02}}
```
