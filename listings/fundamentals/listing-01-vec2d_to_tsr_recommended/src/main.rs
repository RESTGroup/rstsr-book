use rstsr_core::prelude_dev::*;

fn main() -> Result<()> {
    // generate 2-D tensor from 1-D vec, without explicit data copy
    let vec = vec![1, 2, 3, 4, 5, 6];
    let tensor = Tensor::asarray(vec)?.into_shape_assume_contig([2, 3])?;
    println!("{:?}", tensor);
    Ok(())
}
