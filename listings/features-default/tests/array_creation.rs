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
fn example_02() -> Result<()> {
    // ANCHOR: example_02
    use rstsr_core::prelude_dev::*;
    // generate 2-D tensor from 1-D vec, without explicit data copy
    let vec = vec![1, 2, 3, 4, 5, 6];
    let tensor = Tensor::asarray(vec).into_shape_assume_contig([2, 3])?;
    println!("{:?}", tensor);
    // ANCHOR_END: example_02
    Ok(())
}
