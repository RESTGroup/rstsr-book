use rstsr::prelude::*;

#[test]
fn example_elementwise_func() {
    // ANCHOR: example_elem_01
    #[rustfmt::skip]
    let a = rt::asarray(vec![
        5., 2.,
        3., 6.,
        1., 8.,
    ]).into_shape([3, 2]);
    let b = rt::asarray(vec![3., 4.]);

    // broadcasted comparison a >= b
    // called by associated method
    let c = a.greater_equal(&b);

    println!("{:5}", c);
    // output:
    // [[ true  false]
    //  [ true  true ]
    //  [ false true ]]
    // ANCHOR_END: example_elem_01

    // ANCHOR: example_elem_02
    let b = rt::asarray(vec![3., 4.]);
    let d = rt::sin(&b);
    println!("{:6.3}", d);
    // output: [  0.141 -0.757]
    // ANCHOR_END: example_elem_02

    // ANCHOR: example_elem_03
    let a = rt::asarray(vec![0, 2, 5, 7]);
    let b = rt::asarray(vec![1, 3, 4, 6]);

    let c = a.greater_equal(&b);
    println!("{:5}", c);
    // output: [ false false true  true ]

    let c = rt::greater_equal(&a, &b);
    println!("{:5}", c);
    // output: [ false false true  true ]

    let c = rt::ge(&a, &b);
    println!("{:5}", c);
    // output: [ false false true  true ]
    // ANCHOR_END: example_elem_03
}

#[test]
fn example_map() {
    // ANCHOR: example_map_01
    let a: Tensor<f64, _> = rt::asarray(vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]);
    let b = a.mapv(libm::lgamma).mapv(libm::exp);
    println!("{:6.3}", a);
    println!("{:6.3}", b);
    // output:
    // [  0.500  1.000  1.500  2.000  2.500  3.000  3.500  4.000]
    // [  1.772  1.000  0.886  1.000  1.329  2.000  3.323  6.000]

    // please note that in RSTSR, mapv is not lazy evaluated
    // so below code is expected to be more efficient
    let b = a.mapv(|x| libm::exp(libm::lgamma(x)));
    println!("{:6.3}", b);

    // also, function `libm::tgamma` is equivalent to `libm::exp(libm::lgamma(x))`
    // when numerical discrepancy is not a concern
    let b = a.mapv(libm::tgamma);
    println!("{:6.3}", b);
    // ANCHOR_END: example_map_01

    // ANCHOR: example_map_02
    let mut vec_a = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
    let mut a: TensorMut<f64, _> = rt::asarray(&mut vec_a);
    a.mapvi(libm::tgamma);

    // original vector is also modified
    println!("{:6.3?}", vec_a);
    // [ 1.772,  1.000,  0.886,  1.000,  1.329,  2.000,  3.323,  6.000]
    // ANCHOR_END: example_map_02

    // ANCHOR: example_map_03
    #[rustfmt::skip]
    let a = rt::asarray(vec![
        5., 2.,
        3., 6.,
        1., 8.,
    ]).into_shape([3, 2]);
    let b = rt::asarray(vec![3., 4.]);
    let c = a.mapvb(&b, libm::fmin);
    println!("{:6.3}", c);
    // output:
    // [[  3.000  2.000]
    //  [  3.000  4.000]
    //  [  1.000  4.000]]
    // ANCHOR_END: example_map_03
}

#[test]
fn example_reduction() {
    // ANCHOR: example_reduction_01
    #[rustfmt::skip]
    let a = rt::asarray(vec![
        5., 2.,
        3., 6.,
        1., 8.,
    ]).into_shape([3, 2]);

    let b = a.l2_norm();
    println!("{:6.3}", b);
    // output: 11.790

    let b = a.sum_axes(-1);
    println!("{:6.3}", b);
    // output: [  7.000  9.000  9.000]

    let b = a.argmin_axes(0);
    println!("{:6.3}", b);
    // output: [      2      0]
    // ANCHOR_END: example_reduction_01

    // ANCHOR: example_reduction_02
    let a = rt::linspace((-1.0, 1.0, 24)).into_shape([2, 3, 4]);
    let b = a.mean_axes([0, 2]);
    println!("{:6.3}", b);
    // output: [ -0.348 -0.000  0.348]
    // ANCHOR_END: example_reduction_02

    // ANCHOR: example_reduction_03
    let a = rt::asarray(vec![false, true, false, true, true]);
    let b = a.sum();
    println!("{:6}", b);
    // output: 3
    // ANCHOR_END: example_reduction_03
}

#[test]
fn example_linalg() {
    // ANCHOR: example_linalg_01
    let device = DeviceFaer::new(4);
    #[rustfmt::skip]
    let a = rt::asarray((vec![
        1.0, 0.5, 1.5,
        0.5, 5.0, 2.0,
        1.5, 2.0, 8.0,
    ], &device)).into_shape([3, 3]);

    let c = rt::linalg::eigh(&a);
    let (eigenvalues, eigenvectors) = c.into();

    println!("{:8.5}", eigenvalues);
    // [  0.69007  4.01426  9.29567]

    println!("{:8.5}", eigenvectors);
    // [[  0.98056  0.06364 -0.18561]
    //  [ -0.02335 -0.90137 -0.43242]
    //  [ -0.19482  0.42835 -0.88236]]
    // ANCHOR_END: example_linalg_01
}
