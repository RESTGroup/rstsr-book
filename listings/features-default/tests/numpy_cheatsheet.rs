use rstsr::prelude::*;
use rstsr_core::prelude_dev::TensorDeviceChangeAPI;

#[test]
fn example_print_set_max_print() {
    // ANCHOR: example_print_set_max_print
    let a = rt::arange(10);
    println!("{a}");
    // [ 0 1 2 ... 7 8 9]

    use std::sync::atomic::Ordering;
    rstsr_core::format::format_tensor::MAX_PRINT.store(12, Ordering::Relaxed);
    println!("{a}");
    // [ 0 1 2 3 4 5 6 7 8 9]
    // ANCHOR_END: example_print_set_max_print
}

#[test]
fn example_asarray_nested() {
    // ANCHOR: example_asarray_nested
    let device = DeviceFaer::default();
    let l = vec![vec![0, 1, 2], vec![3, 4, 5]];
    let a = rt::asarray((&l, &device));
    println!("{a:?}");
    // [ [0, 1, 2] [3, 4, 5]]
    // 1-Dim (dyn), contiguous: CcFf
    // shape: [2], stride: [1], offset: 0
    // ANCHOR_END: example_asarray_nested

    // ANCHOR: example_asarray_nested_flatten
    let nrows = l.len();
    let ncols = l[0].len(); // may panic if l is empty
    let l = l.iter().flatten().cloned().collect::<Vec<i32>>();
    let a = rt::asarray((&l, [nrows, ncols].c(), &device));
    println!("{a:?}");
    // [[ 0 1 2]
    //  [ 3 4 5]]
    // 2-Dim (dyn), contiguous: Cc
    // shape: [2, 3], stride: [3, 1], offset: 0
    // ANCHOR_END: example_asarray_nested_flatten
}

#[test]
fn example_reshape_order_f() {
    let device = DeviceFaer::default();
    // ANCHOR: example_reshape_order_f_01
    // RSTSR way (1)
    // assuming device.default_order() == RowMajor
    let a = rt::arange((24, &device)).into_shape((2, 3, 4));
    let b = a.t().change_shape((6, 4)).into_reverse_axes();
    // ANCHOR_END: example_reshape_order_f_01
    println!("{b:?}");

    // ANCHOR: example_reshape_order_f_02
    // RSTSR way (2)
    let mut device = a.device().clone();
    device.set_default_order(ColMajor);
    let a = a.into_device(&device);
    let b = a.reshape((4, 6));
    // ANCHOR_END: example_reshape_order_f_02
    println!("{b:?}");
}

#[test]
fn example_blas_dgemm() {
    // ANCHOR: example_blas_dgemm
    let device = DeviceOpenBLAS::default();
    let a = rt::arange((12.0, &device)).into_shape((3, 4)).into_flip(-1);
    let b = rt::arange((12.0, &device)).into_shape((3, 4));
    let mut c = rt::arange((16.0, &device)).into_shape((4, 4));

    use rstsr_blas_traits::blas3::DGEMM;
    let _ = DGEMM::default()
        .a(a.view().into_dim::<Ix2>())
        .b(b.view().into_dim::<Ix2>())
        .c(c.view_mut().into_dim::<Ix2>())
        .alpha(3.0)
        .beta(2.0)
        .transa(Trans)
        .transb(NoTrans)
        .build()
        .unwrap()
        .run();
    println!("{c:?}");
    // ANCHOR_END: example_blas_dgemm
}

#[test]
fn playground() {
    let device = DeviceFaer::default();
    let a = rt::arange((12.0_f64, &device)).into_shape((3, 4));

    let b = a.mapv(|x| x as f32);
    println!("{b:?}");
}
