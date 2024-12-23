use rstsr_core::error::Result;

#[test]
fn example_01() {
    // ANCHOR: example_01
    use rstsr_core::prelude_dev::*;
    // move ownership of vec to 1-D tensor (default CPU device)
    let vec = vec![1, 2, 3, 4, 5];
    let tensor = Tensor::asarray(vec);
    println!("{:?}", tensor);
    // ANCHOR_END: example_01
}

#[test]
fn example_02() {
    // ANCHOR: example_02
    use rstsr_core::prelude_dev::*;
    // move ownership of vec to 1-D tensor
    // custom CPU device that limits threads to 4
    let vec = vec![1, 2, 3, 4, 5];
    let device = DeviceFaer::new(4);
    let tensor = Tensor::asarray((vec, &device));
    println!("{:?}", tensor);
    // ANCHOR_END: example_02
}

#[test]
fn example_03() -> Result<()> {
    // ANCHOR: example_03
    use rstsr_core::prelude_dev::*;
    // generate 2-D tensor from 1-D vec, without explicit data copy
    let vec = vec![1, 2, 3, 4, 5, 6];
    let tensor = Tensor::asarray(vec).into_shape_assume_contig([2, 3]);
    println!("{:?}", tensor);
    // if you feel function `into_shape_assume_contig` ugly, following code also works
    let vec = vec![1, 2, 3, 4, 5, 6];
    let tensor = Tensor::asarray(vec).into_shape([2, 3]).into_owned();
    println!("{:?}", tensor);
    // ANCHOR_END: example_03
    Ok(())
}

#[test]
fn example_04() {
    // ANCHOR: example_04
    use rstsr_core::prelude_dev::*;
    // generate 2-D tensor from nested Vec<T>, WITH EXPLICIT DATA COPY
    let vec = vec![vec![1, 2, 3], vec![4, 5, 6]];
    let tensor = Tensor::asarray(vec.as_ref());
    println!("{:?}", tensor);
    // ANCHOR_END: example_04
}

#[test]
fn example_05() {
    // ANCHOR: example_05
    use rstsr_core::prelude_dev::*;
    // generate 1-D tensor view from &[T], without data copy
    let vec = vec![1, 2, 3, 4, 5, 6];
    let tensor = TensorView::asarray(&vec);
    println!("{:?}", tensor);
    // check if pointer of vec and tensor's storage are the same
    assert_eq!(vec.as_ptr(), tensor.storage().rawvec().as_ptr());
    // ANCHOR_END: example_05
}

#[test]
fn example_06() {
    // ANCHOR: example_06
    use rstsr_core::prelude_dev::*;

    let tensor = Tensor::arange(10);
    println!("{:}", tensor);
    // output: [ 0 1 2 ... 7 8 9]

    let device = DeviceFaer::new(4);
    let tensor = Tensor::arange((2.0, 10.0, &device));
    println!("{:}", tensor);
    // output: [ 2 3 4 5 6 7 8 9]

    let tensor = Tensor::arange((2.0, 3.0, 0.1));
    println!("{:}", tensor);
    // output: [ 2 2.1 2.2 ... 2.7000000000000006 2.8000000000000007 2.900000000000001]
    // ANCHOR_END: example_06
}
