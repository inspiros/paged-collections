# Benchmarks

Run one benchmark group with `cargo bench -p paged-collections-bench --bench sparse` (replace `sparse` with `direct`, `ids`, `seq`, or `pq`). Run all groups with `cargo bench -p paged-collections-bench`. The memory report is printed as CSV to stdout by `cargo run --release -p paged-collections-bench --bin memory`.

For comparable results, run benchmarks on a pinned machine during otherwise idle periods.

Pin the process to dedicated CPUs (for example with `taskset`) and keep the
machine otherwise idle. The largest scattered-key cases can require tens of
GiB, especially direct maps, and Criterion repeats each case. Filter the
benchmarks when exploring on a smaller machine.

Memory rows count allocations requested through the Rust global allocator,
relative to the live allocations before each build; input key vectors are
excluded. The reported peak is the peak tracked live allocation size, not RSS
or allocator overhead. ID, sequence and priority-queue rows use sequential
inputs; the key patterns apply to the sparse and direct groups.
