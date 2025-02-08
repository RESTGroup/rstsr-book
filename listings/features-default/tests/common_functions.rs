use rstsr_core::prelude::*;

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
    // [[ True False]
    //  [ True  True]
    //  [False  True]]

    // sine of b
    // called by function
    let d = rt::sin(&b);

    println!("{:6.3}", d);
    // output: [  0.141 -0.757]
    // ANCHOR_END: example_elem_01

    // ANCHOR: example_elem_02
    use num::complex::c64;

    let a = rt::linspace((c64(1., 2.), c64(-3., 4.), 3));

    // `.abs()` can only called by associated method currently
    let c = a.abs();
    println!("{:8.3}", c);
    // output: [    2.236    3.162    5.000]
    // ANCHOR_END: example_elem_02
}
