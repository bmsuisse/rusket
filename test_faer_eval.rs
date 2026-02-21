use faer::MatRef;

fn main() {
    let factors = vec![1.0_f32, 2.0, 3.0, 4.0];
    let n = 2;
    let k = 2;
    let mat = faer::mat::from_row_major_slice::<f32>(&factors, n, k);
    let tr = mat.transpose();
}
