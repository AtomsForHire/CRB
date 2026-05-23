// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
//
use serde::{Deserialize, Serialize};

use super::SourceComponent;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Source {
    /// The components associated with the source.
    #[serde(with = "serde_yaml::with::singleton_map_recursive")]
    pub components: Box<[SourceComponent]>,
}
