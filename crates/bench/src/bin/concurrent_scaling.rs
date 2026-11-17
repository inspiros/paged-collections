//! Shared-write scaling of the concurrent containers, without Python.
//! Usage: concurrent_scaling [N] [REPEAT] [FILTER]; FILTER keeps the rows
//! whose name contains it.
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use paged_collections::concurrent::{
    AtomicDirectMap, ConcurrentAppendHive, ConcurrentDirectMap, ConcurrentHive,
    ConcurrentIdAllocator, ConcurrentSparseMap,
};

fn permutation(n: usize) -> Vec<u32> {
    let mut keys: Vec<u32> = (0..n as u32).collect();
    let mut s = 0x9E37_79B9_7F4A_7C15_u64;
    for i in (1..n).rev() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        keys.swap(i, (s % (i as u64 + 1)) as usize);
    }
    keys
}

/// Median seconds of `repeat` runs; `setup` builds fresh state outside the timer.
fn time<S: Sync>(
    repeat: usize,
    threads: usize,
    keys: &[u32],
    setup: impl Fn() -> S,
    work: impl Fn(&S, &[u32]) + Sync,
) -> f64 {
    let mut times: Vec<f64> = (0..repeat)
        .map(|_| {
            let state = setup();
            let chunk = keys.len().div_ceil(threads);
            let start = Instant::now();
            std::thread::scope(|scope| {
                for part in keys.chunks(chunk) {
                    let (state, work) = (&state, &work);
                    scope.spawn(move || work(state, part));
                }
            });
            let t = start.elapsed().as_secs_f64();
            drop(black_box(state));
            t
        })
        .collect();
    times.sort_by(f64::total_cmp);
    times[repeat / 2]
}

fn main() {
    let mut args = std::env::args().skip(1);
    let n: usize = args.next().map_or(2_000_000, |a| a.parse().unwrap());
    let repeat: usize = args.next().map_or(5, |a| a.parse().unwrap());
    let filter = args.next().unwrap_or_default();
    let keys = permutation(n);
    let direct_full = || {
        let m = ConcurrentDirectMap::<u32, f64>::new();
        for &k in &keys {
            m.insert(k, 0.0);
        }
        m
    };
    let sparse_full = || {
        let m = ConcurrentSparseMap::<u32, f64>::new();
        for &k in &keys {
            let _ = m.try_insert(k, 0.0);
        }
        m
    };
    println!("N={n}; median of {repeat}; random permutation split into T chunks\n");
    println!("| Workload | T=1 (M/s) | T=2 | T=4 | T=8 | speedup T=4 |");
    println!("|---|---:|---:|---:|---:|---:|");
    let row = |name: &str, f: &dyn Fn(usize) -> f64| {
        if !name.contains(filter.as_str()) {
            return;
        }
        let rates: Vec<f64> = [1, 2, 4, 8]
            .iter()
            .map(|&t| n as f64 / f(t) / 1e6)
            .collect();
        println!(
            "| {name} | {:.1} | {:.1} | {:.1} | {:.1} | {:.2}x |",
            rates[0],
            rates[1],
            rates[2],
            rates[3],
            rates[2] / rates[0]
        );
    };
    row("sparse try_insert (fresh)", &|t| {
        time(
            repeat,
            t,
            &keys,
            ConcurrentSparseMap::<u32, f64>::new,
            |m, ks| {
                for &k in ks {
                    let _ = black_box(m.try_insert(k, 1.0));
                }
            },
        )
    });
    row("sparse try_insert_many (fresh)", &|t| {
        time(
            repeat,
            t,
            &keys,
            ConcurrentSparseMap::<u32, f64>::new,
            |m, ks| {
                black_box(m.try_insert_many(ks.iter().map(|&k| (k, 1.0))));
            },
        )
    });
    row("sparse get", &|t| {
        time(repeat, t, &keys, sparse_full, |m, ks| {
            for &k in ks {
                black_box(m.get(&k));
            }
        })
    });
    row("direct insert (fresh)", &|t| {
        time(
            repeat,
            t,
            &keys,
            ConcurrentDirectMap::<u32, f64>::new,
            |m, ks| {
                for &k in ks {
                    black_box(m.insert(k, 1.0));
                }
            },
        )
    });
    row("direct insert (overwrite)", &|t| {
        time(repeat, t, &keys, direct_full, |m, ks| {
            for &k in ks {
                black_box(m.insert(k, 1.0));
            }
        })
    });
    row("direct get", &|t| {
        time(repeat, t, &keys, direct_full, |m, ks| {
            for &k in ks {
                black_box(m.with(k, |v| *v));
            }
        })
    });
    row("atomic direct insert (fresh)", &|t| {
        time(repeat, t, &keys, AtomicDirectMap::<1024>::new, |m, ks| {
            for &k in ks {
                black_box(m.insert(k as usize, 1));
            }
        })
    });
    row("hive insert (fresh)", &|t| {
        time(repeat, t, &keys, ConcurrentHive::<u64>::new, |h, ks| {
            for &k in ks {
                black_box(h.insert(u64::from(k)));
            }
        })
    });
    row("hive remove", &|t| {
        time(
            repeat,
            t,
            &keys,
            || {
                let h = ConcurrentHive::<u64>::new();
                let ids: Vec<_> = (0..n as u64).map(|v| h.insert(v)).collect();
                (h, ids)
            },
            |(h, ids), ks| {
                for &k in ks {
                    black_box(h.remove(ids[k as usize]));
                }
            },
        )
    });
    row("append hive insert (fresh)", &|t| {
        time(
            repeat,
            t,
            &keys,
            ConcurrentAppendHive::<u64>::new,
            |h, ks| {
                for &k in ks {
                    black_box(h.insert(u64::from(k)));
                }
            },
        )
    });
    row("id allocate (fresh)", &|t| {
        time(
            repeat,
            t,
            &keys,
            ConcurrentIdAllocator::<1024>::new,
            |a, ks| {
                for _ in ks {
                    black_box(a.allocate());
                }
            },
        )
    });
    row("id free", &|t| {
        time(
            repeat,
            t,
            &keys,
            || {
                let a = ConcurrentIdAllocator::<1024>::new();
                let ids: Vec<_> = (0..n).map(|_| a.allocate()).collect();
                (a, ids)
            },
            |(a, ids), ks| {
                for &k in ks {
                    black_box(a.free(ids[k as usize]));
                }
            },
        )
    });
    row("shared counter fetch_add only", &|t| {
        time(
            repeat,
            t,
            &keys,
            || AtomicUsize::new(0),
            |c, ks| {
                for _ in ks {
                    c.fetch_add(1, Ordering::Relaxed);
                }
            },
        )
    });
}
