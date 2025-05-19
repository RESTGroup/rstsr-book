#[test]
#[rustfmt::skip]
fn rstsr_1() {
    // ANCHOR: rstsr_1
    use rstsr::prelude::*;
    use rstsr_openblas::DeviceOpenBLAS;

    // specify the number of threads of 16
    let device = DeviceOpenBLAS::new(16);
    // generate some symmetric matrix
    let a = rt::linspace((-1.0, 1.4, 1048576, &device)).into_shape([1024, 1024]).tan();
    let a = &a + &a.t();
    // evaluate the eigenvalues and eigenvectors
    let (a_eig, a_vec) = rt::linalg::eigh(&a).into();
    println!("eig\n{:10.6}", a_eig);
    println!("vec\n{:10.6}", a_vec);
    // ANCHOR_END: rstsr_1
}

#[test]
#[rustfmt::skip]
fn rstsr_2() {
    // ANCHOR: rstsr_2
    // module rt also exists in rstsr_core, but without linalg support
    use rstsr_core::prelude::*;
    use rstsr_linalg_traits::prelude::rstsr_funcs::eigh;
    use rstsr_openblas::DeviceOpenBLAS;

    // specify the number of threads of 16
    let device = DeviceOpenBLAS::new(16);
    // generate some symmetric matrix
    let a = rt::linspace((-1.0, 1.4, 1048576, &device)).into_shape([1024, 1024]).tan();
    let a = &a + &a.t();
    // evaluate the eigenvalues and eigenvectors
    let (a_eig, a_vec) = eigh(&a).into();
    println!("eig\n{:10.6}", a_eig);
    println!("vec\n{:10.6}", a_vec);
    // ANCHOR_END: rstsr_2
}
