use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::ops::{Deref, DerefMut};

use super::*;

/// Create an index map of the yaml map, each key (source name)
/// has a value of an array of component structs.
///
/// We could instead define a variable SourceList of type IndexMap
/// but then we would have to impl for IndexMap. This method means
/// we can impl for SourceList.

/// The transparent attribute means the whole yaml file is the "value".
/// If it was not here, then the whole yaml file would need a key "source_list"
/// above everything.
/// With this, it means we can save the IndexMap of SourceList to a yaml file
/// without generating a source_list key above everything.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SourceList(IndexMap<String, Source>);

impl SourceList {
    pub(crate) fn new() -> Self {
        return Self::default();
    }
}

impl From<IndexMap<String, Source>> for SourceList {
    fn from(sl: IndexMap<String, Source>) -> Self {
        Self(sl)
    }
}

impl<const N: usize> From<[(String, Source); N]> for SourceList {
    fn from(value: [(String, Source); N]) -> Self {
        Self(IndexMap::from(value))
    }
}

impl Deref for SourceList {
    type Target = IndexMap<String, Source>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for SourceList {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl FromIterator<(String, Source)> for SourceList {
    fn from_iter<I: IntoIterator<Item = (String, Source)>>(iter: I) -> Self {
        let mut c = Self::new();
        for i in iter {
            c.insert(i.0, i.1);
        }
        c
    }
}

impl IntoIterator for SourceList {
    type Item = (String, Source);
    type IntoIter = indexmap::map::IntoIter<String, Source>;

    fn into_iter(self) -> indexmap::map::IntoIter<String, Source> {
        self.0.into_iter()
    }
}
