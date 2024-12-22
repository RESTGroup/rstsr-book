use rstsr_core::prelude_dev::*;

fn main() {
    // move ownership of vec to 1-D tensor, non-default device explicitly specified
    let vec = vec![1, 2, 3, 4, 5];
    // 4 threads available (which is not default; by default all threads available)
    let device = DeviceFaer::new(4);
    let tensor = Tensor::asarray((vec, Some(&device)));
    println!("{:?}", tensor);
}
