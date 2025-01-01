use rstsr_core::prelude::*;

#[test]
fn example_index_by_num() {
    // ANCHOR: example_index_by_num_01
    // generate 3-D tensor A_ijk
    let a = rt::arange(24).into_shape([4, 3, 2]).into_owned();
    println!("{:}", a);

    // B_jk = A_ijk where i = 2
    let b = a.slice(2); // equivalently `a.i(2)`
    println!("{:}", b);
    // output:
    // [[ 12 13]
    //  [ 14 15]
    //  [ 16 17]]
    // ANCHOR_END: example_index_by_num_01

    // ANCHOR: example_index_by_num_02
    // C_k = A_ijk where i = 2, j = 0
    // surely, `a.slice(2).slice(0)` works, but we can use `a.slice([2, 0])` instead
    let c = a.slice([2, 0]);
    println!("{:}", c);
    // output: [ 12 13]
    // ANCHOR_END: example_index_by_num_02

    // ANCHOR: example_index_by_num_03
    // D_jk = A_ijk where i = -1 = 3 (negative index from the end)
    let d = a.slice(-1);
    println!("{:}", d);
    // output:
    // [[ 18 19]
    //  [ 20 21]
    //  [ 22 23]]
    // ANCHOR_END: example_index_by_num_03
}

#[test]
fn example_index_by_range() {
    // ANCHOR: example_index_by_range_01
    // generate 3-D tensor A_ijk
    let a = rt::arange(24).into_shape([4, 3, 2]).into_owned();
    println!("{:}", a);

    // B_ijk = A_ijk where 1 <= i < 3
    let b = a.slice(1..3); // equivalently `a.i(1..3)`
    println!("{:}", b);
    // output:
    // [[[ 6  7]
    //   [ 8  9]
    //   [10 11]]
    //
    //  [[12 13]
    //   [14 15]
    //   [16 17]]]
    // ANCHOR_END: example_index_by_range_01

    // ANCHOR: example_index_by_range_02
    // C_ijk = A_ijk where 1 <= i < 3, 0 <= j < 2
    let c = a.slice([1..3, 0..2]);
    println!("{:}", c);
    // output:
    // [[[ 6  7]
    //   [ 8  9]]
    //
    //  [[12 13]
    //   [14 15]]]
    // ANCHOR_END: example_index_by_range_02

    // ANCHOR: example_index_by_range_03
    let a = rt::arange(24);
    // D_i = A_i where i = -5..-2 = 19..22 (negative index from the end given 24 elements)
    let d = a.slice(-5..-2);
    println!("{:}", d);
    // output: [ 19 20 21]
    // ANCHOR_END: example_index_by_range_03

    // ANCHOR: example_index_by_range_04
    let a = rt::arange(24);
    // D_i = A_i where i = -5.. or 19..
    let d = a.slice(-5..);
    println!("{:}", d);
    // output: [ 19 20 21 22 23]
    // ANCHOR_END: example_index_by_range_04

    // ANCHOR: example_index_by_range_05
    let a = rt::arange(24).into_shape([4, 3, 2]).into_owned();
    let b = a.slice((.., 1..3, ..2)); // equivalently `a.slice(s![.., 1..3, ..2])`
    println!("{:}", b);
    // output:
    // [[[ 2 3]
    //   [ 4 5]]
    //
    //  [[ 8 9]
    //   [ 10 11]]
    //
    //  [[ 14 15]
    //   [ 16 17]]
    //
    //  [[ 20 21]
    //   [ 22 23]]]
    // ANCHOR_END: example_index_by_range_05
}

#[test]
fn example_slice_with_strides() {
    // ANCHOR: example_slice_with_strides_01
    let a = rt::arange(24);

    // first 5 elements
    let b = a.slice(slice!(5));
    println!("{:}", b);
    // output: [ 0 1 2 3 4]

    // elements from 5 to -9 (resembles 15 for the given 24 elements)
    let b = a.slice(slice!(5, -9));
    println!("{:}", b);
    // output: [ 5 6 7 ... 12 13 14]

    // elements from 5 to -9 with step 2
    let b = a.slice(slice!(5, -9, 2));
    println!("{:}", b);
    // output: [ 5 7 9 11 13]

    // reversed step 2
    let b = a.slice(slice!(-9, 5, -2));
    println!("{:}", b);
    // output: [ 15 13 11 9 7]
    // ANCHOR_END: example_slice_with_strides_01

    // ANCHOR: example_slice_with_strides_02
    let b = a.slice(slice!(None, 9, Some(2)));
    println!("{:}", b);
    // output: [ 0 2 4 6 8]
    // ANCHOR_END: example_slice_with_strides_02
}

#[test]
fn example_insert_axes() {
    // ANCHOR: example_insert_axes_01
    let a = rt::arange(24).into_shape([4, 3, 2]).into_owned();

    // insert new axis at the beginning
    let b = a.slice(NewAxis);
    println!("{:?}", b.layout());
    // output: shape: [1, 4, 3, 2], stride: [6, 6, 2, 1], offset: 0

    // using `None` is equivalent to `NewAxis`
    let b = a.slice(None);
    println!("{:?}", b.layout());
    // output: shape: [1, 4, 3, 2], stride: [6, 6, 2, 1], offset: 0

    // insert new axis at the second position
    let b = a.slice((.., None));
    println!("{:?}", b.layout());
    // output: shape: [4, 1, 3, 2], stride: [6, 2, 2, 1], offset: 0
    // ANCHOR_END: example_insert_axes_01
}

#[test]
#[should_panic]
fn example_insert_axes_panic() {
    // ANCHOR: example_insert_axes_02
    let a = rt::arange(24).into_shape([4, 3, 2]).into_owned();

    // insert new axis at the beginning
    let b = a.slice(Some(2));
    println!("{:?}", b.layout());
    // panic: Option<T> should not be used in Indexer.
    // ANCHOR_END: example_insert_axes_02
}

#[test]
fn example_ellipsis() {
    // ANCHOR: example_ellipsis_01
    let a = rt::arange(24).into_shape([4, 3, 2]).into_owned();

    // using ellipsis to select index from last dimension
    // equivallently to `a.slice((.., .., 0))` for 3-D tensor
    // same to numpy's `a[..., 0]`
    let b = a.slice((Ellipsis, 0));
    println!("{:2}", b);
    // output:
    // [[  0  2  4]
    //  [  6  8 10]
    //  [ 12 14 16]
    //  [ 18 20 22]]
    // ANCHOR_END: example_ellipsis_01
}

#[test]
fn example_mixed_indexing() {
    // ANCHOR: example_mixed_indexing
    let a: Tensor<f64, _> = rt::zeros([6, 7, 5, 9, 8]);

    // mixed indexing
    let b = a.slice((slice!(-2, 1, -1), None, None, Ellipsis, 1, ..-2));
    println!("{:?}", b.layout());
    // output: shape: [3, 1, 1, 7, 5, 6], stride: [-2520, 360, 360, 360, 72, 1], offset: 10088
    // ANCHOR_END: example_mixed_indexing
}

#[test]
fn example_elementwise_safe() {
    // ANCHOR: example_elementwise_safe
    let a = rt::arange(24).into_shape([4, 3, 2]).into_owned();

    let val = a[[2, 2, 1]];
    println!("{:}", val);
    // output: 17

    println!("{:}", std::any::type_name_of_val(&val));
    // output: i32
    // ANCHOR_END: example_elementwise_safe

    // ANCHOR: example_elementwise_by_tensor_index
    let view = a.slice((2, 2, 1));
    println!("{:}", view);
    // output: 17

    // it seems to be a value, but actually it is a tensor view
    println!("{:?}", view);
    // output:
    // === Debug Tensor Print ===
    // 17
    // DeviceFaer { base: DeviceCpuRayon { num_threads: 0 } }
    // 0-Dim (dyn), contiguous: CcFf
    // shape: [], stride: [], offset: 17
    // ==========================
    // ANCHOR_END: example_elementwise_by_tensor_index
}

#[test]
#[should_panic]
fn example_elementwise_safe_panic() {
    // ANCHOR: example_elementwise_safe_panic
    let a = rt::arange(24).into_shape([4, 3, 2]).into_owned();

    let val = a[[2, 2, 3]];
    println!("{:}", val);
    // panic: Error::ValueOutOfRange : "idx" = 3 not match to pattern 0..(shp as isize) = 0..2
    // ANCHOR_END: example_elementwise_safe_panic
}

#[test]
fn example_elementwise_unchecked() {
    // ANCHOR: example_elementwise_unchecked
    let a = rt::arange(24).into_shape([4, 3, 2]).into_owned();

    let val = unsafe { a.index_uncheck([2, 2, 1]) };
    println!("{:}", val);
    // output: 17
    // ANCHOR_END: example_elementwise_unchecked
}

#[test]
fn example_elementwise_unchecked_not_desired() {
    // ANCHOR: example_elementwise_unchecked_not_desired
    let a = rt::arange(24).into_shape([4, 3, 2]).into_owned();

    let val = unsafe { a.index_uncheck([2, 2, 3]) };
    println!("{:}", val);
    // output: 19
    // not desired: last dimension index 3 is out of bound
    // ANCHOR_END: example_elementwise_unchecked_not_desired
}
