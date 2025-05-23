use rstsr_core::prelude::*;

#[test]
fn example_tensor_ownership() {
    // ANCHOR: tensor_ownership
    // generate 1-D owned tensor
    let tensor = rt::arange(12);
    let ptr_1 = tensor.as_ptr();

    // this will give owned tensor with 2-D shape
    // since previous tensor is contiguous, this will not copy memory
    let mut tensor = tensor.into_shape([3, 4]);
    tensor += 1; // inplace operation
    let ptr_2 = tensor.as_ptr();

    // until now, memory has not been copied
    assert_eq!(ptr_1, ptr_2);

    // convert to view
    let tensor_view = tensor.view();

    // from view to owned tensor
    let tensor = tensor_view.into_owned();
    let ptr_3 = tensor.as_ptr();

    // now memory has been copied
    assert_ne!(ptr_2, ptr_3);
    // ANCHOR_END: tensor_ownership
}

#[test]
fn example_to_vec() {
    // ANCHOR: to_vec
    // generate *dynamic* 2-D tensor (vec![3, 4] is dynamic)
    let a = rt::arange(12.0).into_shape(vec![3, 4]);
    let b = rt::arange(4.0);

    // matrix multiplication (gemv 2-D x 1-D case)
    let c = a % b;
    println!("{:?}", c);
    let ptr_1 = c.as_ptr();

    // convert to Vec<f64>
    let c = c.into_vec();
    let ptr_2 = c.as_ptr();

    println!("{:?}", c);
    println!("{:?}", core::any::type_name_of_val(&c));
    assert_eq!(c, vec![14.0, 38.0, 62.0]);

    // memory has been moved and no copy occurs if using `into_vec` instead of `to_vec`
    assert_eq!(ptr_1, ptr_2);
    // ANCHOR_END: to_vec

    // ANCHOR: as_ptr_additional
    let a = rt::arange(12.0).into_shape([3, 4]);
    let a_ptr = a.as_ptr();
    println!("{:}", a);
    // output: [[ 0 1 2 3]
    //          [ 4 5 6 7]
    //          [ 8 9 10 11]]

    // please note that b is not contiguous
    // be careful if you want to use `as_ptr` on non-contiguous tensor
    let b = a.i(slice!(1, None, -1));
    let b_ptr = b.as_ptr();
    println!("{:}", b);
    // output: [[ 4 5 6 7]
    //          [ 0 1 2 3]]

    println!("{:}", unsafe { a_ptr.offset_from(b_ptr) });
    // output: -4
    // ANCHOR_END: as_ptr_additional
}

#[test]
fn example_to_scalar() {
    // ANCHOR: to_scalar
    // a: [0, 1, 2, 3, 4]
    let a = rt::arange(5.0);

    // b: [4, 3, 2, 1, 0]
    let b = rt::arange(5.0);
    let b = b.slice(slice!(None, None, -1)); // or b.flip(0)

    // matrix multiplication (dot product for 1-D case)
    let c = a % b;

    // to now, it is still 0-D tensor
    println!("{:?}", c);

    // convert to scalar
    let c = c.to_scalar();
    println!("{:?}", c);
    assert_eq!(c, 10.0);
    // ANCHOR_END: to_scalar
}

#[test]
fn example_dim_conversion() {
    // ANCHOR: dim_conversion
    // fixed dimension
    let a = rt::arange(12).into_shape([3, 4]);
    println!("{:?}", a);
    // output: 2-Dim (dyn), contiguous: Cc

    // convert to dynamic dimension
    let a = a.into_dim::<IxD>(); // or a.into_dyn();
    println!("{:?}", a);
    // output: 2-Dim (dyn), contiguous: Cc

    // convert to fixed dimension again
    let a = a.into_dim::<Ix2>();
    println!("{:?}", a);
    // output: 2-Dim, contiguous: Cc
    // ANCHOR_END: dim_conversion

    // ANCHOR: dyn_dim_construct
    let a = rt::arange(12).into_shape(vec![3, 4]);
    println!("{:?}", a);
    // output: 2-Dim (dyn), contiguous: Cc
    // ANCHOR_END: dyn_dim_construct
}
