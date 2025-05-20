use rstsr::prelude::*;
use rstsr_core::prelude_dev::TensorDeviceChangeAPI;

#[test]
fn example_device_change() {
    // ANCHOR: example_device_change
    let vec = vec![0., 1., 2., 3., 4., 5.];
    let mut device = DeviceFaer::new(4);

    device.set_default_order(RowMajor);
    let a = rt::asarray((&vec, [2, 3], &device));
    println!("{:?}", a);
    // output:
    // [[ 0.0 1.0 2.0]
    //  [ 3.0 4.0 5.0]]
    // 2-Dim (dyn), contiguous: Cc

    device.set_default_order(ColMajor);
    let b = rt::asarray((&vec, [2, 3], &device));
    println!("{:?}", b);
    // output:
    // [[ 0.0 2.0 4.0]
    //  [ 1.0 3.0 5.0]]
    // 2-Dim (dyn), contiguous: Ff
    // ANCHOR_END: example_device_change

    // ANCHOR: example_device_change_to_device
    // change_device will consume variable `b`
    let b_row_major = b.change_device(a.device());
    let c = &a + &b_row_major;
    println!("{:?}", c);
    // output:
    // [[ 0.0 3.0 6.0]
    //  [ 4.0 7.0 10.0]]
    // 2-Dim (dyn), contiguous: Cc
    // ANCHOR_END: example_device_change_to_device
}

#[test]
#[should_panic]
fn example_device_change_panics() {
    let vec = vec![0., 1., 2., 3., 4., 5.];
    let mut device = DeviceFaer::new(4);

    device.set_default_order(RowMajor);
    let a = rt::asarray((&vec, [2, 3], &device));

    device.set_default_order(ColMajor);
    let b = rt::asarray((&vec, [2, 3], &device));

    // ANCHOR: example_device_change_panics
    // a is row-major, b is col-major
    let c = &a + &b;
    // panics:
    // Error::DeviceMismatch : a.device().same_device(b.device())
    // ANCHOR_END: example_device_change_panics
    println!("{:?}", c);
}

#[test]
fn example_reshape() {
    // ANCHOR: example_reshape
    let mut device = DeviceFaer::new(4);
    device.set_default_order(RowMajor);

    let vec_c: Vec<f64> = vec![0., 1., 2., 3., 4., 5.];
    let c = rt::asarray((&vec_c, [2, 3].c(), &device));

    let vec_f: Vec<f64> = vec![0., 3., 1., 4., 2., 5.];
    let f = rt::asarray((&vec_f, [2, 3].f(), &device));

    let diff = (&c - &f).abs().sum();
    println!("{:?}", diff);
    // output: 0.0

    let c_flatten = c.reshape(-1);
    let f_flatten = f.reshape(-1);

    println!("{:}", c_flatten);
    // output: [ 0 1 2 3 4 5]
    println!("{:}", f_flatten);
    // output: [ 0 1 2 3 4 5]

    use core::ptr::eq as ptr_eq;

    // c_flatten shares same ptr to vec_c? true
    println!("{:?}", ptr_eq(vec_c.as_ptr(), c_flatten.raw().as_ptr()));

    // f_flatten shares same ptr to vec_f? false
    println!("{:?}", ptr_eq(vec_f.as_ptr(), f_flatten.raw().as_ptr()));
    // ANCHOR_END: example_reshape

    assert!((diff - 0.0).abs() < 1e-8);
    assert!(ptr_eq(vec_c.as_ptr(), c_flatten.raw().as_ptr()));
    assert!(!ptr_eq(vec_f.as_ptr(), f_flatten.raw().as_ptr()));
}

#[test]
fn example_broadcast_elementwise_01() {
    // ANCHOR: example_broadcast_elementwise_setting_01
    let vec_a = vec![1, 2, 3, 4, 5, 6];
    let a = rt::asarray((&vec_a, [2, 3].c(), &DeviceFaer::default()));
    let vec_b = vec![1, 0, -1];
    let b = rt::asarray((&vec_b, [3], &DeviceFaer::default()));
    // ANCHOR_END: example_broadcast_elementwise_setting_01

    // ANCHOR: example_broadcast_elementwise_01
    let mut device = DeviceFaer::new(4);
    device.set_default_order(RowMajor);

    let a_row_major = a.to_device(&device);
    let b_row_major = b.to_device(&device);

    let c = a_row_major * b_row_major;
    println!("{:3}", c);
    // [[   1   0  -3]
    //  [   4   0  -6]]
    // ANCHOR_END: example_broadcast_elementwise_01
}

#[test]
#[should_panic]
fn example_broadcast_elementwise_fail_01() {
    let vec_a = vec![1, 2, 3, 4, 5, 6];
    let a = rt::asarray((&vec_a, [2, 3].c(), &DeviceFaer::default()));
    let vec_b = vec![1, 0, -1];
    let b = rt::asarray((&vec_b, [3], &DeviceFaer::default()));

    // ANCHOR: example_broadcast_elementwise_fail_01
    let mut device = DeviceFaer::new(4);
    device.set_default_order(ColMajor);

    let a_row_major = a.to_device(&device);
    let b_row_major = b.to_device(&device);

    let c = a_row_major * b_row_major;
    println!("{:3}", c);
    // panics:
    // Error::InvalidLayout : Broadcasting failed. : d1 = 2 not equal to d2 = 3
    // ANCHOR_END: example_broadcast_elementwise_fail_01
}

#[test]
fn example_broadcast_elementwise_02() {
    // ANCHOR: example_broadcast_elementwise_setting_02
    let vec_a = vec![1, 2, 3, 4, 5, 6];
    let a = rt::asarray((&vec_a, [2, 3].c(), &DeviceFaer::default()));
    let vec_b = vec![1, -1];
    let b = rt::asarray((&vec_b, [2], &DeviceFaer::default()));
    // ANCHOR_END: example_broadcast_elementwise_setting_02

    // ANCHOR: example_broadcast_elementwise_02
    let mut device = DeviceFaer::new(4);
    device.set_default_order(ColMajor);

    let a_row_major = a.to_device(&device);
    let b_row_major = b.to_device(&device);

    let c = a_row_major * b_row_major;
    println!("{:3}", c);
    // [[   1   2   3]
    //  [  -4  -5  -6]]
    // ANCHOR_END: example_broadcast_elementwise_02
}

#[test]
#[should_panic]
fn example_broadcast_elementwise_fail_02() {
    let vec_a = vec![1, 2, 3, 4, 5, 6];
    let a = rt::asarray((&vec_a, [2, 3].c(), &DeviceFaer::default()));
    let vec_b = vec![1, -1];
    let b = rt::asarray((&vec_b, [2], &DeviceFaer::default()));

    // ANCHOR: example_broadcast_elementwise_fail_02
    let mut device = DeviceFaer::new(4);
    device.set_default_order(RowMajor);

    let a_row_major = a.to_device(&device);
    let b_row_major = b.to_device(&device);

    let c = a_row_major * b_row_major;
    println!("{:3}", c);
    // panics:
    // Error::InvalidLayout : Broadcasting failed. : d1 = 3 not equal to d2 = 2
    // ANCHOR_END: example_broadcast_elementwise_fail_02
}
