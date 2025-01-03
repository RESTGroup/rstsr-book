use rstsr_core::prelude::*;

#[test]
fn example_basic_arithmetics() {
    // ANCHOR: basic_arithmetics_01
    let a = rt::arange(5.0);
    let b = rt::arange(5.0) + 1.0;

    let c = &a + &b;
    println!("{:}", c);
    // output: [ 1 3 5 7 9]

    let d = &a / &b;
    println!("{:6.3}", d);
    // output: [ 0.000 0.500 0.667 0.750 0.800]

    let e = 2.0 * &a;
    println!("{:}", e);
    // output: [ 0 2 4 6 8]
    // ANCHOR_END: basic_arithmetics_01

    // ANCHOR: basic_arithmetics_02
    let mat = rt::arange(12).into_shape([3, 4]);
    let vec = rt::arange(4).into_shape([4]);

    // matrix multiplication
    let res = &mat % mat.t();
    println!("{:3}", res);
    // output:
    // [[  14  38  62]
    //  [  38 126 214]
    //  [  62 214 366]]

    // matrix-vector multiplication
    let res = &mat % &vec;
    println!("{:}", res);
    // output: [ 14 38 62]

    // vector-matrix multiplication
    let res = &vec % &mat.t();
    println!("{:}", res);
    // output: [ 14 38 62]

    // vector inner dot
    let res = &vec % &vec;
    println!("{:}", res);
    // output: 14
    // ANCHOR_END: basic_arithmetics_02

    // ANCHOR: basic_arithmetics_03
    let a = rt::asarray(vec![true, true, false, false]);
    let b = rt::asarray(vec![true, false, true, false]);

    // bitwise xor
    let c = a ^ b;
    println!("{:?}", c);
    // output: [false true true false]

    let a = rt::asarray(vec![9, 7, 5, 3]);
    let b = rt::asarray(vec![5, 6, 7, 8]);

    // shift left
    let c = a << b;
    println!("{:?}", c);
    // output: [ 288 448 640 768]
    // ANCHOR_END: basic_arithmetics_03
}

#[test]
fn example_op_percent() {
    // ANCHOR: star_as_elem_mult
    let mat = rt::arange(12).into_shape([3, 4]);
    let vec = rt::arange(4);

    // element-wise matrix multiplication
    let c = &mat * &mat;
    println!("{:3}", c);
    // output:
    // [[   0   1   4   9]
    //  [  16  25  36  49]
    //  [  64  81 100 121]]

    // element-wise matrix-vector multiplication (broadcasting involved)
    let d = &mat * &vec;
    println!("{:2}", d);
    // output:
    // [[  0  1  4  9]
    //  [  0  5 12 21]
    //  [  0  9 20 33]]

    // element-wise vector multiplication
    let e = &vec * &vec;
    println!("{:}", e);
    // output: [ 0 1 4 9]
    // ANCHOR_END: star_as_elem_mult

    // ANCHOR: true_rem
    let a = rt::arange(6);

    // remainder to scalar
    let c = rt::rem(&a, 3);
    println!("{:}", c);
    // output: [ 0 1 2 0 1 2]

    // remainder to array
    let b = rt::asarray(vec![3, 2, 3, 3, 2, 2]);
    let c = rt::rem(&a, &b);
    println!("{:}", c);
    // output: [ 0 1 2 0 0 1]
    // ANCHOR_END: true_rem
}

#[test]
fn example_op_percent_confusing() {
    // ANCHOR: confusing_percent_01
    let a = rt::arange(6);
    let b = rt::asarray(vec![3, 2, 3, 3, 2, 2]);

    // remainder to array
    let c = rt::rem(&a, &b);
    println!("{:}", c);
    // output: [ 0 1 2 0 0 1]
    // ANCHOR_END: confusing_percent_01

    // ANCHOR: confusing_percent_02
    // inner product (due to override to `Rem`)
    let c = a.view().rem(&b);
    println!("{:}", c);
    // output: 35
    // ANCHOR_END: confusing_percent_02
}

#[test]
fn example_lt_os_mp2() {
    // ANCHOR: lt_os_mp2_01
    // task definition
    let (naux, nocc, nvir) = (8, 2, 4); // subscripts (P, i, a)
    let y = rt::arange(naux * nocc * nvir).into_shape([naux, nocc, nvir]);
    let ei = rt::arange(nocc);
    let ea = rt::arange(nvir);
    // ANCHOR_END: lt_os_mp2_01

    // ANCHOR: lt_os_mp2_02
    // elementwise multiplication with broadcasting
    // `None` means inserting axis, equivalent to `np.newaxis` in NumPy or `NewAxis` in RSTSR
    let converted_y = &y * ei.slice((None, .., None)) * ea.slice((None, None, ..));
    // ANCHOR_END: lt_os_mp2_02
    println!("{:3}", converted_y);

    // ANCHOR: lt_os_mp2_03
    // elementwise multiplication with simplified broadcasting
    let converted_y = &y * &ei.slice((.., None)) * &ea;
    // ANCHOR_END: lt_os_mp2_03
    println!("{:3}", converted_y);

    // ANCHOR: lt_os_mp2_04
    // optimize for memory access cost
    let converted_y = &y * (&ei.slice((.., None)) * &ea);
    // ANCHOR_END: lt_os_mp2_04
    println!("{:3}", converted_y);
}

#[test]
fn example_ao2mo_vo() {
    // ANCHOR: ao2mo_vo_01
    // task definition
    let (naux, nocc, nvir, nao, _) = (8, 2, 4, 6, 6); // subscripts (P, i, a, μ, ν)
    let y_ao = rt::arange(naux * nao * nao).into_shape([naux, nao, nao]);
    let c_occ = rt::arange(nao * nocc).into_shape([nao, nocc]);
    let c_vir = rt::arange(nao * nvir).into_shape([nao, nvir]);
    // ANCHOR_END: ao2mo_vo_01

    // ANCHOR: ao2mo_vo_02
    let y_mo = &c_occ.t() % &y_ao % &c_vir;
    println!("{:?}", y_mo.layout());
    // ANCHOR_END: ao2mo_vo_02
}

#[test]
fn example_memory_aspects() {
    let a = rt::arange(5.0);
    let b = rt::arange(5.0) + 1.0;

    // ANCHOR: memory_aspects_01
    // arithmetic by reference
    let c = &a + &b;

    // arithmetic by view
    let d = a.view() * b.view();

    // view clone is cheap, given tensor is large
    let a_view = a.view();
    let b_view = b.view();
    let e = a_view.clone() * b_view.clone();
    // ANCHOR_END: memory_aspects_01
    println!("{:}", c);
    println!("{:}", d);
    println!("{:}", e);

    // ANCHOR: memory_aspects_02
    let a = rt::arange(5.0);
    let b = rt::arange(5.0) + 1.0;
    let ptr_a = a.rawvec().as_ptr();
    // if sure that `a` is not used anymore, pass `a` by value instead of reference
    let c = a + &b;
    let ptr_c = c.rawvec().as_ptr();
    // raw data of `a` is reused in `c`
    // similar to `a += &b; let c = a;`
    assert_eq!(ptr_a, ptr_c);
    // ANCHOR_END: memory_aspects_02
}
