use rstsr_core::prelude_dev::*;

fn main() {
    // move ownership of vec to 1-D tensor (default CPU device)
    let vec = vec![1, 2, 3, 4, 5];
    let tensor = Tensor::asarray(vec);
    println!("{:?}", tensor);
}
