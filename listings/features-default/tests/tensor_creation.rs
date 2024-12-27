#[test]
fn example_01() {
    use rstsr_core::prelude::rstsr as rt;

    // ANCHOR: example_01
    // move ownership of vec to 1-D tensor (default CPU device)
    let vec = vec![1.0, 2.968, 3.789, 4.35, 5.575];
    let tensor = rt::asarray(vec);

    // only print 2 decimal places
    println!("{:.2}", tensor);
    // output: [ 1.00 2.97 3.79 4.35 5.58]
    // ANCHOR_END: example_01
}

#[test]
fn example_02() {
    use rstsr_core::prelude::rstsr as rt;
    use rt::DeviceFaer;

    // ANCHOR: example_02
    // move ownership of vec to 1-D tensor
    // custom CPU device that limits threads to 4
    let vec = vec![1, 2, 3, 4, 5];
    let device = DeviceFaer::new(4);
    let tensor = rt::asarray((vec, &device));
    println!("{:?}", tensor);

    // output:
    // === Debug Tensor Print ===
    // [ 1 2 3 4 5]
    // DeviceFaer { base: DeviceCpuRayon { num_threads: 4 } }
    // 1-Dim, contiguous: CcFf
    // shape: [5], stride: [1], offset: 0
    // Type: rstsr_core::tensorbase::TensorBase<rstsr_core::tensor::data::DataOwned<rstsr_core::storage::device::Storage<i32, rstsr_core::device_faer::device::DeviceFaer>>, [usize; 1]>
    // ANCHOR_END: example_02
}

#[test]
fn example_03() {
    use rstsr_core::prelude::rstsr as rt;

    // ANCHOR: example_03
    // generate 2-D tensor from 1-D vec, without explicit data copy
    let vec = vec![1, 2, 3, 4, 5, 6];
    let tensor = rt::asarray(vec).into_shape_assume_contig([2, 3]);
    println!("{:}", tensor);

    // if you feel function `into_shape_assume_contig` ugly, following code also works
    let vec = vec![1, 2, 3, 4, 5, 6];
    let tensor = rt::asarray(vec).into_shape([2, 3]).into_owned();
    println!("{:}", tensor);

    // and even more concise
    let vec = vec![1, 2, 3, 4, 5, 6];
    let tensor = rt::asarray((vec, [2, 3]));
    println!("{:}", tensor);

    // output:
    // [[ 1 2 3]
    //  [ 4 5 6]]
    // ANCHOR_END: example_03
}

#[test]
fn example_04() {
    use rstsr_core::prelude::rstsr as rt;
    use rstsr_core::prelude::*;

    // ANCHOR: example_04
    let vec = vec![vec![1, 2, 3], vec![4, 5, 6]];

    // generate 2-D tensor from nested Vec<T>, WITH EXPLICIT DATA COPY
    // so this is not recommended for large data
    let (nrow, ncol) = (vec.len(), vec[0].len());
    let vec = vec.into_iter().flatten().collect::<Vec<_>>();

    // please also note that nested vec is always row-major, so using `.c()` is more appropriate
    let tensor = rt::asarray((vec, [nrow, ncol].c()));
    println!("{:}", tensor);
    // output:
    // [[ 1 2 3]
    //  [ 4 5 6]]
    // ANCHOR_END: example_04
}

#[test]
fn example_05() {
    use rstsr_core::prelude::rstsr as rt;

    // ANCHOR: example_05
    // generate 1-D tensor view from &[T], without data copy
    let vec = vec![1, 2, 3, 4, 5, 6];
    let tensor = rt::asarray(&vec);

    // note `tensor` is TensorView instead of Tensor, so it doesn't own data
    println!("{:?}", tensor);

    // check if pointer of vec and tensor's storage are the same
    assert_eq!(vec.as_ptr(), tensor.storage().rawvec().as_ptr());

    // output:
    // === Debug Tensor Print ===
    // [ 1 2 3 4 5 6]
    // DeviceFaer { base: DeviceCpuRayon { num_threads: 0 } }
    // 1-Dim, contiguous: CcFf
    // shape: [6], stride: [1], offset: 0
    // Type: rstsr_core::tensorbase::TensorBase<rstsr_core::tensor::data::DataRef<rstsr_core::storage::device::Storage<i32, rstsr_core::device_faer::device::DeviceFaer>>, [usize; 1]>
    // ANCHOR_END: example_05
}

#[test]
fn example_06() {
    use rstsr_core::prelude::rstsr as rt;

    // ANCHOR: example_06
    // generate 2-D tensor mutable view from &mut [T], without data copy
    let mut vec = vec![1, 2, 3, 4, 5, 6];
    let mut tensor = rt::asarray((&mut vec, [2, 3]));

    // you may perform arithmetic operations on `tensor`
    tensor *= 2;
    println!("{:}", tensor);
    // output:
    // [[ 2 4 6]
    //  [ 8 10 12]]

    // you may also see variable `vec` is also changed
    println!("{:?}", vec);
    // output: [2, 4, 6, 8, 10, 12]
    // ANCHOR_END: example_06
}

#[test]
fn example_arange() {
    use rstsr_core::prelude::rstsr as rt;
    use rt::DeviceFaer;

    // ANCHOR: example_arange
    let tensor = rt::arange(10);
    println!("{:}", tensor);
    // output: [ 0 1 2 ... 7 8 9]

    let device = DeviceFaer::new(4);
    let tensor = rt::arange((2.0, 10.0, &device));
    println!("{:}", tensor);
    // output: [ 2 3 4 5 6 7 8 9]

    let tensor = rt::arange((2.0, 3.0, 0.1));
    println!("{:}", tensor);
    // output: [ 2 2.1 2.2 ... 2.7000000000000006 2.8000000000000007 2.900000000000001]
    // ANCHOR_END: example_arange
}

#[test]
fn example_linspace() {
    use rstsr_core::prelude::rstsr as rt;
    use rt::DeviceFaer;

    // ANCHOR: example_linspace
    use num::complex::c64;

    let tensor = rt::linspace((0.0, 10.0, 11));
    println!("{:}", tensor);
    // output: [ 0 1 2 ... 8 9 10]

    let tensor = rt::linspace((c64(1.0, 2.0), c64(-15.0, 10.0), 5, &DeviceFaer::new(4)));
    println!("{:}", tensor);
    // output: [ 1+2i -3+4i -7+6i -11+8i -15+10i]
    // ANCHOR_END: example_linspace
}

#[test]
fn example_eye() {
    use rstsr_core::prelude::rstsr as rt;
    use rt::{DeviceFaer, Tensor};

    // ANCHOR: example_eye
    let device = DeviceFaer::new(4);
    let tensor: Tensor<f64, _> = rt::eye((3, &device));
    println!("{:}", tensor);
    // output:
    // [[ 1 0 0]
    //  [ 0 1 0]
    //  [ 0 0 1]]

    let tensor: Tensor<f64, _> = rt::eye((3, 4, -1));
    println!("{:}", tensor);
    // output:
    // [[ 0 0 0 0]
    //  [ 1 0 0 0]
    //  [ 0 1 0 0]]
    // ANCHOR_END: example_eye
}

#[test]
fn example_diag() {
    use rstsr_core::prelude::rstsr as rt;
    use rstsr_core::prelude::*;

    // ANCHOR: example_diag
    let vec = rt::arange(3) + 1;
    let tensor = vec.diag();
    println!("{:}", tensor);
    // output:
    // [[ 1 0 0]
    //  [ 0 2 0]
    //  [ 0 0 3]]

    let tensor = rt::arange(9).into_shape([3, 3]).into_owned();
    let diag = tensor.diag();
    println!("{:}", diag);
    // output: [ 0 4 8]
    // ANCHOR_END: example_diag
}

#[test]
fn example_zeros_01() {
    use rstsr_core::prelude::rstsr as rt;
    use rt::{DeviceCpuSerial, Tensor};

    // ANCHOR: example_zeros_01
    // generate tensor with default device
    let tensor: Tensor<f64, _> = rt::zeros([2, 2, 3]); // Tensor<f64, Ix3>
    println!("{:}", tensor);
    // output:
    // [[[ 0 0 0]
    //   [ 0 0 0]]
    //
    //  [[ 0 0 0]
    //   [ 0 0 0]]]

    // generate tensor with custom device
    // note: the third type annotation refers to device type, hence is required if not default device
    // Tensor<f64, Ix2, DeviceCpuSerial>
    let tensor: Tensor<f64, _, _> = rt::zeros(([3, 4], &DeviceCpuSerial));
    println!("{:}", tensor);
    // output:
    // [[ 0 0 0 0]
    //  [ 0 0 0 0]
    //  [ 0 0 0 0]]
    // ANCHOR_END: example_zeros_01
}

#[test]
fn example_zeros_02() {
    use rstsr_core::prelude::rstsr as rt;
    use rstsr_core::prelude::*;
    use rt::Tensor;

    // ANCHOR: example_zeros_02
    // generate tensor with c-contiguous
    let tensor: Tensor<f64, _> = rt::zeros([2, 2, 3].c());
    println!("shape: {:?}, stride: {:?}", tensor.shape(), tensor.stride());
    // output: shape: [2, 2, 3], stride: [6, 3, 1]

    // generate tensor with f-contiguous
    let tensor: Tensor<f64, _> = rt::zeros([2, 2, 3].f());
    println!("shape: {:?}, stride: {:?}", tensor.shape(), tensor.stride());
    // output: shape: [2, 2, 3], stride: [1, 2, 4]
    // ANCHOR_END: example_zeros_02
}

#[test]
fn example_zeros_03() {
    use rstsr_core::prelude::rstsr as rt;
    use rt::Tensor;

    // ANCHOR: example_zeros_03
    // generate 0-D tensor
    let mut a: Tensor<f64, _> = rt::zeros([]);
    println!("{:}", a);
    // output: 0

    // 0-D tensor arithmetics are also valid
    a += 2.0;
    println!("{:}", a);
    // output: 2

    let b = rt::arange(3.0) + 1.0;
    let c = a + b;
    println!("{:}", c);
    // output: [ 3 4 5]
    // ANCHOR_END: example_zeros_03
}

#[test]
fn example_empty() {
    use rstsr_core::prelude::rstsr as rt;
    use rt::Tensor;

    // ANCHOR: example_empty
    // generate empty tensor with default device
    let tensor: Tensor<i32, _> = unsafe { rt::empty([10, 10]) };
    println!("{:?}", tensor);
    // ANCHOR_END: example_empty
}

#[test]
fn example_random() {
    use rstsr_core::prelude::rstsr as rt;
    use rstsr_core::prelude::*;

    // ANCHOR: example_random
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    // generate f-contiguous layout and it's memory buffer size
    let layout = [2, 3].f();
    let size = layout.size();

    // generate random numbers to vector
    let seed: u64 = 42;
    let mut rng = StdRng::seed_from_u64(seed);
    let random_vec: Vec<f64> = (0..size).map(|_| rng.gen()).collect();

    // create a tensor from random vector and f-contiguous layout
    let tensor = rt::asarray((random_vec, layout));

    // print tensor with 3 decimal places with width of 7
    println!("{:7.3}", tensor);
    // ANCHOR_END: example_random
}
