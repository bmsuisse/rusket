use faer::Mat;

fn main() {
    let mut a = Mat::<f32>::zeros(10, 10);
    let mut b = Mat::<f32>::zeros(10, 1);
    
    // faer 0.24 Cholesky API
    use faer::prelude::*;
    let llt = a.as_ref().cholesky(faer::Side::Lower).unwrap();
    llt.solve_in_place(b.as_mut());
}
