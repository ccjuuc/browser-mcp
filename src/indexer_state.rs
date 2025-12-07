use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileState {
    pub last_modified: u64,
    pub size: u64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct IndexerState {
    pub file_states: HashMap<String, FileState>,
}

impl IndexerState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        if !path.as_ref().exists() {
            return Ok(Self::default());
        }

        let content = fs::read_to_string(path)
            .context("Failed to read indexer state file")?;
        
        // Handle empty file case
        if content.trim().is_empty() {
            return Ok(Self::default());
        }

        let state = serde_json::from_str(&content)
            .context("Failed to parse indexer state JSON")?;
        
        Ok(state)
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = serde_json::to_string_pretty(self)
            .context("Failed to serialize indexer state")?;
        
        fs::write(path, content)
            .context("Failed to write indexer state file")?;
        
        Ok(())
    }

    pub fn should_index(&self, file_path: &str, current_modified: u64, current_size: u64) -> bool {
        if let Some(state) = self.file_states.get(file_path) {
            // Index if modified time or size has changed
            state.last_modified != current_modified || state.size != current_size
        } else {
            // New file
            true
        }
    }

    pub fn update(&mut self, file_path: String, last_modified: u64, size: u64) {
        self.file_states.insert(
            file_path,
            FileState {
                last_modified,
                size,
            },
        );
    }
}
