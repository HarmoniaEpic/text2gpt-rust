use anyhow::Result;
use candle_core::{Device, Tensor};
use rand::seq::SliceRandom;
use rand::thread_rng;

use crate::tokenizer::GPT2Tokenizer;

/// Text dataset for training
pub struct TextDataset {
    tokens: Vec<i64>,
    max_length: usize,
    device: Device,
    pad_token_id: i64,
}

impl TextDataset {
    /// Create a new dataset from text samples
    pub fn new(texts: &[String], tokenizer: &GPT2Tokenizer, max_length: usize, device: &Device) -> Result<Self> {
        let mut all_tokens = Vec::new();
        
        // Tokenize all texts
        for text in texts {
            let encoded = tokenizer.encode(text)?;
            all_tokens.extend(encoded.iter().map(|&x| x as i64));
        }
        
        log::info!("Created dataset with {} tokens", all_tokens.len());
        
        Ok(Self {
            tokens: all_tokens,
            max_length,
            device: device.clone(),
            pad_token_id: tokenizer.pad_token_id() as i64,
        })
    }
    
    /// Get the number of samples in the dataset
    pub fn len(&self) -> usize {
        if self.tokens.len() > self.max_length {
            self.tokens.len() - self.max_length
        } else {
            1
        }
    }
    
    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Get a sample at the given index
    pub fn get(&self, idx: usize) -> Result<(Tensor, Tensor)> {
        let chunk: Vec<i64> = if idx + self.max_length + 1 <= self.tokens.len() {
            self.tokens[idx..idx + self.max_length + 1].to_vec()
        } else {
            // Pad if necessary
            let mut chunk = self.tokens[idx..].to_vec();
            let pad_len = self.max_length + 1 - chunk.len();
            chunk.extend(vec![self.pad_token_id; pad_len]);
            chunk
        };
        
        let x = Tensor::from_slice(&chunk[..self.max_length], (self.max_length,), &self.device)?;
        let y = Tensor::from_slice(&chunk[1..], (self.max_length,), &self.device)?;
        
        Ok((x, y))
    }
    
    /// Create a dataloader with the specified batch size
    pub fn dataloader(&self, batch_size: usize, shuffle: bool) -> DataLoader {
        DataLoader::new(self, batch_size, shuffle)
    }
}

/// Enhanced dataloader with better shuffling
pub struct DataLoader<'a> {
    dataset: &'a TextDataset,
    batch_size: usize,
    indices: Vec<usize>,
    current_idx: usize,
    shuffle: bool,
}

impl<'a> DataLoader<'a> {
    fn new(dataset: &'a TextDataset, batch_size: usize, shuffle: bool) -> Self {
        let indices: Vec<usize> = (0..dataset.len()).collect();
        
        let mut loader = Self {
            dataset,
            batch_size,
            indices,
            current_idx: 0,
            shuffle,
        };
        
        if shuffle {
            loader.shuffle_indices();
        }
        
        loader
    }
    
    /// Shuffle the indices
    fn shuffle_indices(&mut self) {
        self.indices.shuffle(&mut thread_rng());
    }
    
    /// Get the number of batches
    pub fn len(&self) -> usize {
        (self.dataset.len() + self.batch_size - 1) / self.batch_size
    }
    
    /// Reset the dataloader
    pub fn reset(&mut self, shuffle: bool) {
        self.current_idx = 0;
        if shuffle || self.shuffle {
            self.shuffle_indices();
        }
    }
    
    /// Check if dataloader is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<'a> Iterator for DataLoader<'a> {
    type Item = Result<(Tensor, Tensor)>;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.indices.len() {
            return None;
        }
        
        let batch_end = (self.current_idx + self.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.current_idx..batch_end];
        
        let mut x_batch = Vec::new();
        let mut y_batch = Vec::new();
        
        for &idx in batch_indices {
            match self.dataset.get(idx) {
                Ok((x, y)) => {
                    x_batch.push(x);
                    y_batch.push(y);
                }
                Err(e) => return Some(Err(e)),
            }
        }
        
        self.current_idx = batch_end;
        
        // Stack into batch tensors
        match (Tensor::stack(&x_batch, 0), Tensor::stack(&y_batch, 0)) {
            (Ok(x), Ok(y)) => Some(Ok((x, y))),
            (Err(e), _) | (_, Err(e)) => Some(Err(e.into())),
        }
    }
}