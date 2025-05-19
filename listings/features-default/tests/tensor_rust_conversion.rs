use rstsr::prelude::*;

#[test]
fn example_from_vec_by_asarray() {
    // ANCHOR: example_from_vec_by_asarray
    let device = DeviceOpenBLAS::new(16);
    let vec = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = rt::asarray((vec, [2, 3], &device));
    println!("{tensor:8.4}");
    // output (row-major):
    // [[   1.0000   2.0000   3.0000]
    //  [   4.0000   5.0000   6.0000]]
    // output (col-major):
    // [[   1.0000   3.0000   5.0000]
    //  [   2.0000   4.0000   6.0000]]
    // ANCHOR_END: example_from_vec_by_asarray
}

#[test]
fn example_from_vec_by_scratch() {
    // ANCHOR: example_from_vec_by_scratch
    use rstsr_core::prelude_dev::*;

    // step 1: wrap vector into data representation
    let vec = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let data: DataOwned<Vec<f64>> = DataOwned::from(vec);

    // step 2: construct device
    let device: DeviceOpenBLAS = DeviceOpenBLAS::new(16);

    // step 3: construct storage that pinned to device
    let storage: Storage<DataOwned<Vec<f64>>, f64, DeviceOpenBLAS> = Storage::new(data, device);

    // step 4: construct layout (row-major case that last stride is 1)
    // this will give 2-D layout with dynamic shape
    // arguments: ([nrow, ncol], [stride_row, stride_col], offset)
    let layout = Layout::new(vec![2, 3], vec![3, 1], 0).unwrap();
    // if you insist to use static shape, you can use:
    // let layout = Layout::new([2, 3], [3, 1], 0).unwrap();

    // step 5: construct tensor
    let tensor = Tensor::new(storage, layout);

    println!("{tensor:8.4}");
    // output:
    // [[   1.0000   2.0000   3.0000]
    //  [   4.0000   5.0000   6.0000]]
    // ANCHOR_END: example_from_vec_by_scratch
}

#[test]
#[should_panic]
fn example_into_vec_by_into_vec_failed() {
    let device = DeviceOpenBLAS::new(16);
    let vec = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = rt::asarray((vec, [2, 3], &device));

    // ANCHOR: example_into_vec_by_into_vec_failed
    println!("{:?}", tensor.shape());
    // output: [2, 3]
    let vec = tensor.into_vec();
    println!("{:?}", vec);
    // ANCHOR_END: example_into_vec_by_into_vec_failed
}

#[test]
fn example_into_vec_by_into_vec() {
    // ANCHOR: example_into_vec_by_into_vec
    // create tensor
    let vec_raw = vec![1, 2, 3, 4, 5, 6];
    let ptr_raw = vec_raw.as_ptr();
    let tensor = rt::asarray(vec_raw);

    // convert tensor to vector
    let vec_out = tensor.into_vec();
    let ptr_out = vec_out.as_ptr();
    println!("{vec_out:?}");

    // data is moved and no copy occurs
    assert_eq!(ptr_raw, ptr_out);
    // ANCHOR_END: example_into_vec_by_into_vec

    // ANCHOR: example_into_vec_by_into_vec_cloned
    // create tensor with stride -1
    // by flip the tensor along the 0-th axis
    let vec_raw = vec![1, 2, 3, 4, 5, 6];
    let ptr_raw = vec_raw.as_ptr();
    let tensor = rt::asarray(vec_raw).into_flip(0);
    println!("{tensor:?}");
    // output: [6, 5, 4, 3, 2, 1]

    // convert tensor to vector
    let vec_out = tensor.into_vec();
    let ptr_out = vec_out.as_ptr();
    println!("{vec_out:?}");
    // output: [6, 5, 4, 3, 2, 1]

    // data is cloned, so this `into_vec` is expensive
    assert_ne!(ptr_raw, ptr_out);
    // ANCHOR_END: example_into_vec_by_into_vec_cloned
}

#[test]
fn example_into_vec_destruct() {
    // ANCHOR: example_into_vec_destruct
    // construct tensor by asarray
    let device = DeviceOpenBLAS::new(16);
    let vec = vec![1, 2, 3, 4, 5, 6];
    let tensor = rt::asarray((vec, [2, 3], &device));

    // step 1: tensor -> (storage, layout)
    let (storage, layout) = tensor.into_raw_parts();
    println!("{layout:?}");

    // step 2: storage -> (data, device)
    let (data, device) = storage.into_raw_parts();
    println!("{device:?}");

    // step 3: data -> raw, where DeviceOpenBLAS::Raw = Vec<T>
    let vec = data.into_raw();
    println!("{vec:?}");
    // output: [1, 2, 3, 4, 5, 6]
    // ANCHOR_END: example_into_vec_destruct
}

#[test]
fn example_into_vec_destruct_warn() {
    // ANCHOR: example_into_vec_destruct_warn
    let vec_raw = vec![1, 2, 3, 4, 5, 6];
    let ptr_raw = vec_raw.as_ptr();
    let tensor = rt::asarray(vec_raw).into_flip(0);

    // step 1: tensor -> (storage, layout)
    let (storage, layout) = tensor.into_raw_parts();
    println!("{layout:?}");
    // output:
    // 1-Dim (dyn), contiguous: Custom
    // shape: [6], stride: [-1], offset: 5

    // step 2: storage -> (data, device)
    let (data, device) = storage.into_raw_parts();
    println!("{device:?}");

    // step 3: data -> raw, where DeviceOpenBLAS::Raw = Vec<T>
    // in this way, original `vec` will be returned
    let vec_out = data.into_raw();
    let ptr_out = vec_out.as_ptr();
    assert_eq!(ptr_raw, ptr_out);
    println!("{vec_out:?}");
    // output: [1, 2, 3, 4, 5, 6]

    // please note that, `tensor.into_vec` will give
    // output: [6, 5, 4, 3, 2, 1]
    // ANCHOR_END: example_into_vec_destruct_warn
}

#[test]
fn example_from_ref_by_asarray() {
    // ANCHOR: example_from_ref_by_asarray
    let device = DeviceOpenBLAS::new(16);
    let vec = vec![1, 2, 3, 4, 5, 6];
    let tensor = rt::asarray((&vec, [2, 3], &device));
    println!("{tensor}");
    // output (row-major):
    // [[ 1 2 3]
    //  [ 4 5 6]]
    // output (col-major):
    // [[ 1 3 5]
    //  [ 2 4 6]]
    // ANCHOR_END: example_from_ref_by_asarray
}

#[test]
fn example_from_ref_by_scratch() {
    // ANCHOR: example_from_ref_by_scratch
    use rstsr_core::prelude_dev::*;
    use std::mem::ManuallyDrop;

    let vec = vec![1, 2, 3, 4, 5, 6];
    let vec_ref: &[usize] = &vec;

    // step 1: wrap reference into data representation
    // this uses `ManuallyDrop` to avoid double free
    let vec_manual_drop: ManuallyDrop<Vec<usize>> = ManuallyDrop::new(unsafe {
        Vec::from_raw_parts(vec_ref.as_ptr() as *mut _, vec_ref.len(), vec_ref.len())
    });
    let data: DataRef<Vec<usize>> = DataRef::ManuallyDropOwned(vec_manual_drop);

    // step 2: construct device
    let device: DeviceOpenBLAS = DeviceOpenBLAS::new(16);

    // step 3: construct storage that pinned to device
    let storage: Storage<DataRef<Vec<usize>>, usize, DeviceOpenBLAS> = Storage::new(data, device);

    // step 4: construct layout (row-major case that last stride is 1)
    // this will give 2-D layout with dynamic shape
    // arguments: ([nrow, ncol], [stride_row, stride_col], offset)
    let layout = Layout::new(vec![2, 3], vec![3, 1], 0).unwrap();
    // if you insist to use static shape, you can use:
    // let layout = Layout::new([2, 3], [3, 1], 0).unwrap();

    // step 5: construct tensor
    let tensor = TensorView::new(storage, layout);

    println!("{tensor}");
    // output:
    // [[ 1 2 3]
    //  [ 4 5 6]]
    // ANCHOR_END: example_from_ref_by_scratch
}

#[test]
fn example_to_ref_by_raw() {
    // ANCHOR: example_to_ref_by_raw
    let vec = vec![1, 2, 3, 4, 5, 6];
    let tensor = rt::asarray((vec, [2, 3]));
    println!("{tensor}");
    // output (row-major):
    // [[ 1 2 3]
    //  [ 4 5 6]]

    let slc: &Vec<usize> = tensor.raw();
    println!("{slc:?}");
    // output: [1, 2, 3, 4, 5, 6]
    // ANCHOR_END: example_to_ref_by_raw
}

#[test]
fn example_raw_error_usage() {
    // ANCHOR: example_raw_error_usage_1
    // prepare tensor
    let vec: Vec<f64> = vec![1.0, 0.5, 2.0, 0.5, 5.0, 1.5, 2.0, 1.5, 8.0];
    let device = DeviceFaer::default();
    let tensor = rt::asarray((vec.clone(), [3, 3].f(), &device));
    println!("{tensor:8.4}");
    // output:
    // [[   1.0000   0.5000   2.0000]
    //  [   0.5000   5.0000   1.5000]
    //  [   2.0000   1.5000   8.0000]]
    // ANCHOR_END: example_raw_error_usage_1

    // ANCHOR: example_raw_error_usage_2
    // standard way to perform Cholesky by RSTSR
    let sub_mat = tensor.i((1..3, 1..3));
    // [[   5.0000   1.5000]
    //  [   1.5000   8.0000]]
    let sub_chol = rt::linalg::cholesky((&sub_mat, Lower));
    println!("{sub_chol:8.4}");
    // [[   2.2361   0.0000]
    //  [   0.6708   2.7477]]
    // ANCHOR_END: example_raw_error_usage_2

    // ANCHOR: example_raw_error_usage_3
    // wrong way to perform Cholesky by LAPACK
    let mut tensor = rt::asarray((vec.clone(), [3, 3].f(), &device));
    let mut sub_mat = tensor.i_mut((1..3, 1..3));
    let mut info = 0;
    unsafe { lapack::dpotrf(b'L', 2, sub_mat.raw_mut(), 3, &mut info) };
    println!("{sub_mat:8.4}");
    // This is not what we want! The `sub_mat.raw_mut()` points to 1.0 instead of 5.0!
    // It actually diagonalizes tensor[0:2, 0:2] instead of tensor[1:3, 1:3]!
    // [[   2.1794   1.5000]
    //  [   1.5000   8.0000]]
    // ANCHOR_END: example_raw_error_usage_3
    assert_eq!(info, 0);

    // ANCHOR: example_raw_error_usage_4
    // correct way to perform Cholesky by LAPACK
    let mut tensor = rt::asarray((vec.clone(), [3, 3].f(), &device));
    let mut sub_mat = tensor.i_mut((1..3, 1..3));
    let offset = sub_mat.offset(); // offset is 4
    let mut info = 0;
    // notice that we need to add offset to the pointer
    unsafe { lapack::dpotrf(b'L', 2, &mut sub_mat.raw_mut()[offset..], 3, &mut info) };
    println!("{sub_mat:8.4}");
    // [[   2.2361   1.5000]    upper-triangular does not matter
    //  [   0.6708   2.7477]]
    // ANCHOR_END: example_raw_error_usage_4
    assert_eq!(info, 0);
}

#[test]
fn example_rstsr_faer_conversion() {
    // ANCHOR: example_rstsr_faer_conversion
    let vec: Vec<i32> = vec![1, 2, 3, 4, 5, 6];
    let device = DeviceFaer::default();
    let tensor = rt::asarray((vec, [2, 3], &device));
    println!("{tensor}");
    // [[ 1 2 3]
    //  [ 4 5 6]]

    // convert to faer tensor
    use faer_ext::IntoFaer;
    let faer_tensor = tensor.view().into_dim::<Ix2>().into_faer();
    println!("{faer_tensor:?}");
    // [
    // [1, 2, 3],
    // [4, 5, 6],
    // ]

    // convert back to rstsr tensor
    use rstsr_core::tensor::ext_conversion::IntoRSTSR;
    let rstsr_tensor = faer_tensor.into_rstsr();
    println!("{rstsr_tensor}");
    // [[ 1 2 3]
    //  [ 4 5 6]]
    // ANCHOR_END: example_rstsr_faer_conversion
}
