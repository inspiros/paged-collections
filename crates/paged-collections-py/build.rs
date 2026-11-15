//! Generate the finite set of const-generic storage variants used by Python.

use std::{env, fs, path::PathBuf};

fn size(value: &str, variable: &str) -> usize {
    let size = value.trim().parse::<usize>().unwrap_or_else(|_| {
        panic!("{variable}: expected an integer in 1..=1048576, got {value:?}")
    });
    assert!(
        (1..=1 << 20).contains(&size),
        "{variable}: page size must be in 1..=1048576, got {size}"
    );
    size
}

fn main() {
    println!("cargo:rerun-if-env-changed=PAGED_COLLECTIONS_PAGE_SIZES");
    println!("cargo:rerun-if-env-changed=PAGED_COLLECTIONS_DEFAULT_PAGE_SIZE");
    println!("cargo:rerun-if-changed=build.rs");
    let configured = env::var("PAGED_COLLECTIONS_PAGE_SIZES")
        .unwrap_or_else(|_| "256,512,1024,2048,4096".to_owned());
    assert!(
        !configured.trim().is_empty(),
        "PAGED_COLLECTIONS_PAGE_SIZES: the list must not be empty"
    );
    let mut sizes: Vec<_> = configured
        .split(',')
        .map(|s| size(s, "PAGED_COLLECTIONS_PAGE_SIZES"))
        .collect();
    sizes.sort_unstable();
    assert!(
        sizes.windows(2).all(|pair| pair[0] != pair[1]),
        "PAGED_COLLECTIONS_PAGE_SIZES: duplicate page sizes are not allowed"
    );
    let default = size(
        &env::var("PAGED_COLLECTIONS_DEFAULT_PAGE_SIZE").unwrap_or_else(|_| "1024".to_owned()),
        "PAGED_COLLECTIONS_DEFAULT_PAGE_SIZE",
    );
    assert!(
        sizes.contains(&default),
        "PAGED_COLLECTIONS_DEFAULT_PAGE_SIZE: {default} must be in PAGED_COLLECTIONS_PAGE_SIZES"
    );

    // Brackets separate declaration/use generics and type arguments before/after
    // PAGE. This also accepts nested types and an optional directory argument.
    let variants = sizes
        .iter()
        .map(|n| format!("P{n}($inner<$($before,)* {n} $(, $after)*>),"))
        .collect::<String>();
    let constructors = sizes
        .iter()
        .map(|n| format!("{n} => Ok(Self::P{n}($inner::new())),"))
        .collect::<String>();
    let page_sizes = sizes
        .iter()
        .map(|n| format!("Self::P{n}(_) => {n},"))
        .collect::<String>();
    let mut code = format!(
        "/// Compiled page sizes, sorted ascending.\npub(crate) const PAGE_SIZES: &[usize] = &{sizes:?};\n\
         /// Default page size for new Python containers.\npub(crate) const DEFAULT_PAGE_SIZE: usize = {default};\n"
    );
    code.push_str(&format!(r#"
macro_rules! paged_enum {{
    ($vis:vis enum $name:ident [$($decl:tt)*] [$($arg:ty),*] =
        $inner:ident [$($before:ty),*] [$($after:ty),*]) => {{
        /// Storage with a page size selected at construction.
        #[derive(Clone)]
        $vis enum $name<$($decl)*> {{ {variants} }}
        impl<$($decl)*> $name<$($arg),*> {{
            pub(crate) fn new(page_size: usize) -> pyo3::PyResult<Self> {{
                match page_size {{ {constructors} _ => Err($crate::page_size_error(page_size)), }}
            }}
            pub(crate) fn page_size(&self) -> usize {{ match self {{ {page_sizes} }} }}
        }}
        impl<$($decl)*> Default for $name<$($arg),*> {{
            fn default() -> Self {{ Self::new($crate::DEFAULT_PAGE_SIZE).expect("compiled default page size") }}
        }}
    }};
}}
"#));
    for (name, iterator) in [("paged", false), ("paged_iter", true)] {
        code.push_str(&format!("macro_rules! {name} {{ ($enum:ident, $value:expr, $binding:ident => $body:expr) => {{ match $value {{"));
        for n in &sizes {
            let body = if iterator {
                format!("$crate::PageIter::P{n}($body)")
            } else {
                "$body".to_owned()
            };
            code.push_str(&format!("$enum::P{n}($binding) => {body},"));
        }
        code.push_str("} }; }\n");
    }
    code.push_str("macro_rules! paged_wrap { ($enum:ident, $value:expr, $binding:ident => $body:expr) => { match $value {");
    for n in &sizes {
        code.push_str(&format!("$enum::P{n}($binding) => $body.map($enum::P{n}),"));
    }
    code.push_str("} }; }\n");
    // A sum of iterator types preserves static dispatch and iterator guarantees
    // without boxing, snapshots, or a size-by-size cross product.
    let params = (0..sizes.len())
        .map(|i| format!("I{i}"))
        .collect::<Vec<_>>();
    let args = params.join(",");
    code.push_str(&format!("enum PageIter<{args}> {{"));
    for (n, p) in sizes.iter().zip(&params) {
        code.push_str(&format!("P{n}({p}),"));
    }
    code.push_str("}\n");
    for (bound, trait_name, methods) in [
        (
            "Iterator",
            "Iterator",
            vec![
                ("next", "Option<Self::Item>"),
                ("size_hint", "(usize, Option<usize>)"),
            ],
        ),
        ("ExactSizeIterator", "ExactSizeIterator", vec![]),
        (
            "DoubleEndedIterator",
            "DoubleEndedIterator",
            vec![("next_back", "Option<Self::Item>")],
        ),
    ] {
        let bounds = params
            .iter()
            .enumerate()
            .map(|(i, p)| {
                if i == 0 {
                    format!("{p}: {bound}")
                } else {
                    format!("{p}: {bound}<Item = I0::Item>")
                }
            })
            .collect::<Vec<_>>()
            .join(",");
        code.push_str(&format!(
            "impl<{bounds}> {trait_name} for PageIter<{args}> {{"
        ));
        if trait_name == "Iterator" {
            code.push_str("type Item = I0::Item;");
        }
        for (method, result) in methods {
            let receiver = if method == "size_hint" {
                "&self"
            } else {
                "&mut self"
            };
            code.push_str(&format!(
                "fn {method}({receiver}) -> {result} {{ match self {{"
            ));
            for n in &sizes {
                code.push_str(&format!("Self::P{n}(iter) => iter.{method}(),"));
            }
            code.push_str("} }");
        }
        code.push_str("}\n");
    }
    fs::write(
        PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR")).join("page_sizes.rs"),
        code,
    )
    .expect("write generated page_sizes.rs");
}
