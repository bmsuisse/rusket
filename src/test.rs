use faer::prelude::*;
fn test() {
    let mut a: faer::Mat<f64> = faer::Mat::zeros(10, 10);
    let mut b: faer::Mat<f64> = faer::Mat::zeros(10, 1);
    
    let llt = a.as_ref().cholesky(faer::Side::Lower).unwrap();
    llt.solve_in_place(b.as_mut());
}
