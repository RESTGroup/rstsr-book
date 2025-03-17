use rstsr_core::prelude::*;

#[test]
fn asarray_override() {
    // ANCHOR: asarray_override
    // Tensor<usize, DeviceCpu> (owned tensor on default device)
    let a = rt::asarray(vec![0usize, 1, 2]);

    // Tensor<usize, DeviceFaer> (owned tensor with Faer backend and 4 threads)
    // note the double parentheses here
    let device = DeviceFaer::new(4);
    let b = rt::asarray((vec![0usize, 1, 2], &device));

    // TensorView<usize, DeviceCpu> (view tensor on default device)
    // no data copy is performed when initializing this tensor view
    let vec_c = vec![0usize, 1, 2, 3];
    let c = rt::asarray(&vec_c);
    // c: [0, 1, 2, 3]

    // TensorView<usize, DeviceCpu> with 2x2 shape
    let d = rt::asarray((&vec_c, [2, 2]));
    // d: [[ 0 1]
    //     [ 2 3]]
    // ANCHOR_END: asarray_override
    println!("{:?}", a);
    println!("{:?}", b);
    println!("{:?}", c);
    println!("{:?}", d);
}
