//! Shared visitor for maps that validate each entry before inserting it.

use core::fmt;
use core::marker::PhantomData;
use serde::de::{Deserialize, Deserializer, Error, MapAccess, Visitor};

pub(super) fn deserialize_map<'de, De, T, K, V>(
    deserializer: De,
    output: T,
    insert: impl FnMut(&mut T, K, V) -> Result<(), &'static str>,
) -> Result<T, De::Error>
where
    De: Deserializer<'de>,
    K: Deserialize<'de>,
    V: Deserialize<'de>,
{
    struct MapVisitor<T, K, V, F> {
        output: T,
        insert: F,
        entry: PhantomData<fn(K, V)>,
    }

    impl<'de, T, K, V, F> Visitor<'de> for MapVisitor<T, K, V, F>
    where
        K: Deserialize<'de>,
        V: Deserialize<'de>,
        F: FnMut(&mut T, K, V) -> Result<(), &'static str>,
    {
        type Value = T;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("a map of keys to values")
        }

        fn visit_map<A: MapAccess<'de>>(mut self, mut access: A) -> Result<T, A::Error> {
            while let Some((key, value)) = access.next_entry()? {
                (self.insert)(&mut self.output, key, value).map_err(A::Error::custom)?;
            }
            Ok(self.output)
        }
    }

    deserializer.deserialize_map(MapVisitor {
        output,
        insert,
        entry: PhantomData,
    })
}
