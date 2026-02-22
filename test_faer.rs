use faer::Mat;

fn main() {
    let mut a = faer::Mat::<f64>::zeros(10, 10);
    let mut b = faer::Mat::<f64>::zeros(10, 1);
    
    // faer 0.21.2 llt formulation
    
    // allocate factorization 
    let par = faer::Par::Seq;
    let req = faer::linalg::cholesky::llt::factor::cholesky_in_place_scratch::<f64>(
        10,
        par,
        Default::default(),
    );
    let mut mem = faer::dyn_stack::MemBuffer::new(req);
    let mut stack = faer::dyn_stack::MemStack::new(&mut mem);
    
    // solve
    if let Ok(_) = faer::linalg::cholesky::llt::factor::cholesky_in_place(
        a.as_mut(),
        faer::Par::Seq,
        stack,
        Default::default(),
    ) {
    
    }
}
