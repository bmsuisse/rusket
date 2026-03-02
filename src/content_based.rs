use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;

use ahash::AHashMap;
use std::collections::BTreeMap;

/// Tokenize a string into whitespace-separated lowercase tokens, then produce
/// n-grams in the given range. Simple stop-word removal for English.
fn tokenize(text: &str, ngram_min: usize, ngram_max: usize) -> Vec<String> {
    static STOP_WORDS: &[&str] = &[
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can", "need", "dare",
        "ought", "used", "not", "no", "nor", "so", "if", "than", "too",
        "very", "just", "about", "above", "after", "again", "all", "also",
        "am", "any", "because", "before", "between", "both", "each",
        "few", "further", "get", "got", "her", "here", "him", "his", "how",
        "i", "into", "it", "its", "me", "more", "most", "my", "new",
        "now", "only", "other", "our", "out", "over", "own", "s", "same",
        "she", "some", "still", "such", "t", "that", "their", "them",
        "then", "there", "these", "they", "this", "those", "through",
        "under", "up", "us", "we", "what", "when", "where", "which",
        "while", "who", "whom", "why", "you", "your",
    ];

    // Simple word tokenization: lowercase, split on non-alphanumeric
    let words: Vec<String> = text
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 1 && !STOP_WORDS.contains(&w.as_ref()))
        .map(|w| w.to_string())
        .collect();

    let mut ngrams = Vec::new();
    for n in ngram_min..=ngram_max {
        if n > words.len() {
            continue;
        }
        for i in 0..=(words.len() - n) {
            ngrams.push(words[i..i + n].join(" "));
        }
    }

    ngrams
}

/// Compute TF-IDF vectors and pairwise cosine similarity matrix.
///
/// Returns a flat f32 array of shape (n_docs × n_docs) with the diagonal zeroed.
fn tfidf_cosine_sim(
    texts: &[String],
    max_features: usize,
    ngram_min: usize,
    ngram_max: usize,
) -> Vec<f32> {
    let n_docs = texts.len();

    // Step 1: Tokenize all documents and build vocabulary with DF counts
    let doc_tokens: Vec<Vec<String>> = texts
        .iter()
        .map(|t| tokenize(t, ngram_min, ngram_max))
        .collect();

    // Count document frequency for each term
    let mut df: AHashMap<String, usize> = AHashMap::new();
    for tokens in &doc_tokens {
        let mut seen = ahash::AHashSet::new();
        for tok in tokens {
            if seen.insert(tok.clone()) {
                *df.entry(tok.clone()).or_insert(0) += 1;
            }
        }
    }

    // Select top-max_features terms by DF (use BTreeMap for deterministic ordering)
    let mut terms_by_df: Vec<(String, usize)> = df.into_iter().collect();
    terms_by_df.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    terms_by_df.truncate(max_features);

    // Use BTreeMap for deterministic feature ordering
    let vocab: BTreeMap<String, usize> = terms_by_df
        .iter()
        .enumerate()
        .map(|(idx, (term, _))| (term.clone(), idx))
        .collect();
    let n_features = vocab.len();

    if n_features == 0 {
        // No features: return zero similarity
        return vec![0.0f32; n_docs * n_docs];
    }

    // Step 2: Compute TF-IDF matrix (n_docs × n_features), row-major
    let n_docs_f64 = n_docs as f64;
    let mut tfidf = vec![0.0f64; n_docs * n_features];

    for (doc_idx, tokens) in doc_tokens.iter().enumerate() {
        // Term frequency
        let mut tf: AHashMap<usize, f64> = AHashMap::new();
        for tok in tokens {
            if let Some(&feat_idx) = vocab.get(tok) {
                *tf.entry(feat_idx).or_insert(0.0) += 1.0;
            }
        }

        let row = &mut tfidf[doc_idx * n_features..(doc_idx + 1) * n_features];

        for (&feat_idx, &count) in &tf {
            // IDF: log((1 + n) / (1 + df)) + 1 (sklearn's smooth IDF)
            let doc_freq = terms_by_df[feat_idx].1 as f64;
            let idf = ((1.0 + n_docs_f64) / (1.0 + doc_freq)).ln() + 1.0;
            row[feat_idx] = count * idf;
        }
    }

    // L2 normalize each document vector (for cosine similarity)
    for doc_idx in 0..n_docs {
        let row = &mut tfidf[doc_idx * n_features..(doc_idx + 1) * n_features];
        let norm: f64 = row.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 0.0 {
            for v in row.iter_mut() {
                *v /= norm;
            }
        }
    }

    // Step 3: Compute pairwise cosine similarity (dot products of L2-normalized vectors)
    let mut sim = vec![0.0f32; n_docs * n_docs];
    sim.par_chunks_mut(n_docs)
        .enumerate()
        .for_each(|(i, sim_row)| {
            let row_i = &tfidf[i * n_features..(i + 1) * n_features];
            for j in 0..n_docs {
                if i == j {
                    sim_row[j] = 0.0; // zero diagonal
                } else {
                    let row_j = &tfidf[j * n_features..(j + 1) * n_features];
                    let mut dot = 0.0f64;
                    for k in 0..n_features {
                        dot += row_i[k] * row_j[k];
                    }
                    sim_row[j] = dot as f32;
                }
            }
        });

    sim
}

#[pyfunction]
#[pyo3(signature = (texts, max_features, ngram_min, ngram_max))]
pub fn tfidf_cosine_similarity<'py>(
    py: Python<'py>,
    texts: Vec<String>,
    max_features: usize,
    ngram_min: usize,
    ngram_max: usize,
) -> PyResult<Py<PyArray2<f32>>> {
    let n_docs = texts.len();

    let sim = py.detach(|| {
        tfidf_cosine_sim(&texts, max_features, ngram_min, ngram_max)
    });

    let arr = PyArray1::from_vec(py, sim);
    Ok(arr.reshape([n_docs, n_docs])?.into())
}
