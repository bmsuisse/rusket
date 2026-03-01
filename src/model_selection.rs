use pyo3::prelude::*;
use rand::prelude::*;
use std::collections::HashMap;

#[pyfunction]
#[pyo3(signature = (user_ids, item_ids, timestamps=None))]
pub fn leave_one_out(
    user_ids: Vec<i32>,
    item_ids: Vec<i32>,
    timestamps: Option<Vec<f32>>,
) -> (Vec<usize>, Vec<usize>) {
    let n = user_ids.len();
    if n != item_ids.len() {
        panic!("User and item arrays must have the same length.");
    }
    
    if let Some(ref ts) = timestamps {
        if n != ts.len() {
             panic!("Timestamps array must have the same length as user and item arrays.");
        }
    }

    // Group interactions by user
    let mut user_interactions: HashMap<i32, Vec<usize>> = HashMap::new();
    for i in 0..n {
        user_interactions.entry(user_ids[i]).or_default().push(i);
    }

    let mut train_indices = Vec::new();
    let mut test_indices = Vec::new();

    let mut rng = rand::rng();

    for (_user, indices) in user_interactions {
        if indices.is_empty() {
             continue;
        }
        
        if indices.len() == 1 {
            // If only one interaction, it must go to train (cannot evaluate on nothing)
            train_indices.push(indices[0]);
            continue;
        }

        let test_idx = if let Some(ref ts) = timestamps {
            // Find the index with the maximum timestamp
            let mut max_t_idx = indices[0];
            let mut max_t = ts[max_t_idx];
            for &idx in indices.iter().skip(1) {
                if ts[idx] > max_t {
                    max_t = ts[idx];
                    max_t_idx = idx;
                }
            }
            max_t_idx
        } else {
            // Randomly select one index for the test set
            *indices.choose(&mut rng).unwrap()
        };

        // Add to train/test
        test_indices.push(test_idx);
        for &idx in &indices {
            if idx != test_idx {
                train_indices.push(idx);
            }
        }
    }

    (train_indices, test_indices)
}

#[pyfunction]
pub fn train_test_split(user_ids: Vec<i32>, test_ratio: f32) -> (Vec<usize>, Vec<usize>) {
    let n = user_ids.len();
    if test_ratio < 0.0 || test_ratio > 1.0 {
        panic!("test_ratio must be between 0 and 1.");
    }

    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng = rand::rng();
    indices.shuffle(&mut rng);

    let test_size = (n as f32 * test_ratio).round() as usize;

    let test_indices: Vec<usize> = indices.iter().take(test_size).copied().collect();
    let train_indices: Vec<usize> = indices.iter().skip(test_size).copied().collect();

    (train_indices, test_indices)
}
