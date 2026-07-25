//! A pool of N interchangeable "lanes", where **holding the lane IS holding the
//! right to use it** — so the two can never drift apart.
//!
//! # The defect this exists to make impossible
//!
//! The previous design kept a `Semaphore` with N permits beside a
//! `Mutex<Vec<Session>>` with N sessions, and stated the invariant *in a doc
//! comment*: "sem has exactly N permits, so a permit-holder is always guaranteed
//! a session to pop". Two independent things, one prose promise. It broke in two
//! ways, both observed in production (the 2026-07-11 → 07-18 outage):
//!
//! 1. **A lane leaked on every abnormal exit.** The session was moved into a
//!    `spawn_blocking` closure and handed back *only* by the success return. A
//!    panic dropped it; so did **caller cancellation** — a client disconnect or
//!    read timeout — because a detached blocking task still runs to completion
//!    and its return value is simply discarded. The second path needs no panic
//!    at all and leaves no log line.
//! 2. **Then the pool mutex was poisoned, permanently.** Once sessions < permits,
//!    a permit-holder popped `None` and panicked *while the `MutexGuard`
//!    temporary was still alive*, so unwinding poisoned the `std::sync::Mutex`.
//!    Every later `lock().unwrap()` panicked. The embed path was 100% dead while
//!    the process stayed up and `/health` reported "healthy" for seven days.
//!
//! # How this type removes both, by construction rather than by comment
//!
//! * The lane lives in a **bounded channel**. Receiving one *is* acquiring it;
//!   returning it *is* releasing it. There is no second counter to drift from.
//! * There is **no `std::sync::Mutex`**, so poisoning is not expressible. The
//!   receiver sits behind a `tokio::sync::Mutex`, which has no poisoning.
//! * [`Lane`] returns itself in **`Drop`**. When the guard is moved into the
//!   blocking closure, `Drop` therefore runs on **success, panic, and caller
//!   cancellation alike** — because a blocking task always runs to completion.
//!
//! # Generic over the lane on purpose
//!
//! Production instantiates this with `ort::session::Session`, which can only be
//! built from 547 MB of ONNX weights that CI does not have. The invariant above
//! is the thing that broke, so the invariant is what must be tested — the tests
//! at the bottom of this file drive a pool of trivial dummy lanes and are
//! weights-free and CI-safe.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tokio::sync::{mpsc, Mutex};

/// A borrowed lane that returns itself to the pool when dropped.
///
/// Move this **into** the blocking closure that uses it. That placement is
/// load-bearing: a blocking task cannot be cancelled, so once the guard is
/// inside, its `Drop` is guaranteed to run no matter how the caller exits.
pub struct Lane<T> {
    /// `Some` for the whole life of the guard; `None` only inside `Drop`.
    item: Option<T>,
    tx: mpsc::Sender<T>,
    lost: Arc<AtomicUsize>,
    /// Always `true` in production. Test-only pools built with
    /// [`LanePool::new_legacy`] set this `false` to reproduce the pre-fix policy
    /// ("returned only by the caller, on the success path"), so the defect this
    /// module removes stays provable forever instead of only in a build log.
    return_on_drop: bool,
}

impl<T> Lane<T> {
    /// The pooled value. Infallible in practice: `item` is only taken by `Drop`,
    /// and the borrow checker forbids calling this on a dropped guard.
    pub fn get_mut(&mut self) -> &mut T {
        self.item
            .as_mut()
            .expect("lane item is present for the whole life of the guard")
    }
}

impl<T> Drop for Lane<T> {
    fn drop(&mut self) {
        let Some(item) = self.item.take() else {
            return;
        };
        if !self.return_on_drop {
            // Legacy policy, tests only: the lane is NOT returned here, exactly
            // as before the fix — where the push happened in the async caller
            // after the join, and so never ran on a panic or a cancellation.
            return;
        }
        // ⚠️ THIS FUNCTION MUST NEVER PANIC.
        //
        // `Drop` runs during unwinding, and a panic *while already unwinding*
        // aborts the whole process — which would replace a poisoned mutex with a
        // hard crash: the same class of defect, louder. So the return is a
        // `try_send` whose error branch logs and drops, and never unwraps.
        //
        // `try_send` cannot actually be `Full`: the channel's capacity equals the
        // lane count, and at most that many lanes exist to be returned. The
        // branch is unreachable by construction — and is still handled loudly
        // rather than asserted away, because "impossible" is exactly what the
        // last doc comment claimed.
        if let Err(err) = self.tx.try_send(item) {
            self.lost.fetch_add(1, Ordering::SeqCst);
            tracing::error!(
                error = %err,
                "lane could NOT be returned to the pool — capacity permanently reduced"
            );
        }
    }
}

/// A fixed set of lanes, handed out one at a time.
pub struct LanePool<T> {
    tx: mpsc::Sender<T>,
    rx: Mutex<mpsc::Receiver<T>>,
    capacity: usize,
    lost: Arc<AtomicUsize>,
    return_on_drop: bool,
}

impl<T> LanePool<T> {
    /// Build a pool from the lanes it will hand out.
    ///
    /// # Panics
    /// If `items` is empty. This is a boot-time configuration error (a pool with
    /// no lanes can never serve a request), consistent with the other
    /// fail-at-startup checks in `main`.
    pub fn new(items: Vec<T>) -> Self {
        Self::build(items, true)
    }

    /// The **pre-fix** policy, for tests only: a lane is returned only by an
    /// explicit success-path release, never by `Drop`. Reproduces the design that
    /// caused the 2026-07-11 outage so the regression stays provable.
    #[cfg(test)]
    pub fn new_legacy(items: Vec<T>) -> Self {
        Self::build(items, false)
    }

    fn build(items: Vec<T>, return_on_drop: bool) -> Self {
        assert!(!items.is_empty(), "lane pool must have at least one lane");
        let capacity = items.len();
        let (tx, rx) = mpsc::channel(capacity);
        for item in items {
            // Cannot fail: we just sized the channel to exactly this many items.
            tx.try_send(item)
                .unwrap_or_else(|_| unreachable!("channel sized to the lane count"));
        }
        Self {
            tx,
            rx: Mutex::new(rx),
            capacity,
            lost: Arc::new(AtomicUsize::new(0)),
            return_on_drop,
        }
    }

    /// Lanes the pool was built with.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Lanes permanently lost. Should always be 0; a non-zero value means the
    /// unreachable branch in [`Lane::drop`] fired and is worth alerting on.
    pub fn lost(&self) -> usize {
        self.lost.load(Ordering::SeqCst)
    }

    /// Lanes that can still ever serve a request (busy ones included).
    pub fn available_capacity(&self) -> usize {
        self.capacity.saturating_sub(self.lost())
    }

    /// Take a lane, waiting for a busy one to come back.
    ///
    /// Returns `None` only when every lane has been permanently lost — the
    /// honest "this can never succeed" answer, so the caller can refuse with a
    /// 503 instead of waiting forever on a queue that will never move.
    ///
    /// Cancel-safe: `recv` does not consume a lane unless it completes, so a
    /// caller that gives up while waiting here takes nothing with it.
    pub async fn acquire(&self) -> Option<Lane<T>> {
        if self.available_capacity() == 0 {
            return None;
        }
        let item = self.rx.lock().await.recv().await?;
        Some(Lane {
            item: Some(item),
            tx: self.tx.clone(),
            lost: Arc::clone(&self.lost),
            return_on_drop: self.return_on_drop,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// Stand-in for a `Session`: cheap, weights-free, and counts its own death.
    struct DummyLane {
        #[allow(dead_code)]
        id: usize,
    }

    fn pool_of(n: usize) -> Arc<LanePool<DummyLane>> {
        Arc::new(LanePool::new((0..n).map(|id| DummyLane { id }).collect()))
    }

    /// Baseline: a normal request gives the lane back.
    #[tokio::test]
    async fn success_returns_the_lane() {
        let pool = pool_of(2);
        {
            let mut lane = pool.acquire().await.expect("lane available");
            let _ = lane.get_mut();
        }
        assert_eq!(pool.available_capacity(), 2);
        assert_eq!(pool.lost(), 0);
        // All lanes really are re-acquirable.
        assert!(can_serve(&pool, 2).await, "both lanes re-acquirable");
    }

    /// STAGE 1, panic half — a panicking inference must not consume a lane.
    ///
    /// RED against the old design: the session was moved into the closure and
    /// returned only on the success path, so a panic destroyed it.
    #[tokio::test]
    async fn panic_in_blocking_closure_returns_the_lane() {
        let pool = pool_of(2);
        let lane = pool.acquire().await.expect("lane available");

        let joined = tokio::task::spawn_blocking(move || {
            // The guard is moved in, so unwinding runs its Drop.
            let mut lane = lane;
            let _ = lane.get_mut();
            panic!("onnx inference failed");
        })
        .await;

        assert!(joined.is_err(), "the blocking task must report its panic");
        assert_eq!(
            pool.available_capacity(),
            2,
            "a panic must not shrink the pool"
        );
        assert_eq!(pool.lost(), 0);

        // The pool still serves both lanes afterwards — the real proof.
        // Uses the timeout helper deliberately: a starved pool does not error,
        // it waits forever, so a bare `acquire().await` would make a REGRESSION
        // HANG THE SUITE instead of failing it. Measured, not guessed — an
        // earlier run of this file against the pre-fix policy hung for 10
        // minutes rather than reporting a failure.
        assert!(
            can_serve(&pool, 2).await,
            "both lanes must still be servable after a panic"
        );
    }

    /// STAGE 1, cancellation half (FINDING 1) — an abandoned caller must not
    /// consume a lane, and this happens with **no panic at all**.
    ///
    /// RED against the old design: a detached blocking task ran to completion and
    /// its `(vectors, session)` was discarded, silently shrinking the pool. This
    /// is the routine client-timeout path, not an exceptional one.
    #[tokio::test]
    async fn cancelled_caller_does_not_leak_a_lane() {
        let pool = pool_of(2);

        let work = {
            let pool = Arc::clone(&pool);
            async move {
                let lane = pool.acquire().await.expect("lane available");
                tokio::task::spawn_blocking(move || {
                    let mut lane = lane;
                    let _ = lane.get_mut();
                    std::thread::sleep(Duration::from_millis(300));
                })
                .await
                .expect("blocking task")
            }
        };

        // The client gives up long before the work finishes.
        let outcome = tokio::time::timeout(Duration::from_millis(50), work).await;
        assert!(outcome.is_err(), "the caller must have timed out");

        // Let the detached blocking task finish and drop its guard.
        tokio::time::sleep(Duration::from_millis(500)).await;

        assert_eq!(
            pool.available_capacity(),
            2,
            "a cancelled caller must not shrink the pool"
        );
        assert_eq!(pool.lost(), 0);

        assert!(
            can_serve(&pool, 2).await,
            "both lanes must still be servable after a cancelled caller"
        );
    }

    /// Repeated abuse must not accumulate damage — the old design died after two.
    #[tokio::test]
    async fn repeated_panics_never_exhaust_the_pool() {
        let pool = pool_of(2);
        for _ in 0..10 {
            let lane = pool.acquire().await.expect("lane available");
            let _ = tokio::task::spawn_blocking(move || {
                // Moved in on purpose: its Drop must run while unwinding.
                let _lane = lane;
                panic!("onnx inference failed");
            })
            .await;
        }
        assert_eq!(pool.available_capacity(), 2);
        assert_eq!(pool.lost(), 0);
        assert!(can_serve(&pool, 2).await, "pool still serves both lanes");
    }

    /// Can this pool still hand out `n` lanes, or has it starved?
    ///
    /// A starved pool doesn't error — it *waits forever* on a queue that will
    /// never move, which is precisely why the outage was invisible. So starvation
    /// is detected by a short timeout, and the lanes are held (not dropped) until
    /// the count is taken.
    async fn can_serve(pool: &Arc<LanePool<DummyLane>>, n: usize) -> bool {
        let mut held = Vec::new();
        for _ in 0..n {
            match tokio::time::timeout(Duration::from_millis(100), pool.acquire()).await {
                Ok(Some(lane)) => held.push(lane),
                _ => return false,
            }
        }
        true
    }

    // ---- The pre-fix policy, pinned as permanent regressions. -------------
    // These assert the OLD behaviour and must keep passing: they are the "red"
    // for the two tests above, preserved in the suite instead of only in a build
    // log. If a future change makes them fail, the defect class is back.

    /// PRE-FIX: a panic destroyed the lane. Two panics killed the shipped
    /// default pool (`EMBEDDING_POOL_SIZE` defaults to 2).
    #[tokio::test]
    async fn legacy_policy_loses_a_lane_on_panic() {
        let pool = Arc::new(LanePool::new_legacy(
            (0..2).map(|id| DummyLane { id }).collect(),
        ));
        for _ in 0..2 {
            let lane = pool.acquire().await.expect("lane available");
            let _ = tokio::task::spawn_blocking(move || {
                // Moved in on purpose: its Drop must run while unwinding.
                let _lane = lane;
                panic!("onnx inference failed");
            })
            .await;
        }
        assert!(
            !can_serve(&pool, 1).await,
            "PRE-FIX pool must be starved after 2 panics — this is the defect"
        );
    }

    /// PRE-FIX, and the one that needs no panic at all (FINDING 1): a client that
    /// times out silently took a lane with it.
    #[tokio::test]
    async fn legacy_policy_loses_a_lane_on_cancellation() {
        let pool = Arc::new(LanePool::new_legacy(
            (0..2).map(|id| DummyLane { id }).collect(),
        ));
        for _ in 0..2 {
            let pool2 = Arc::clone(&pool);
            let work = async move {
                let lane = pool2.acquire().await.expect("lane available");
                tokio::task::spawn_blocking(move || {
                    // Moved in on purpose: its Drop runs when the task ends.
                    let _lane = lane;
                    std::thread::sleep(Duration::from_millis(300));
                })
                .await
                .expect("blocking task")
            };
            assert!(
                tokio::time::timeout(Duration::from_millis(50), work)
                    .await
                    .is_err(),
                "caller must have timed out"
            );
        }
        tokio::time::sleep(Duration::from_millis(600)).await;
        assert!(
            !can_serve(&pool, 1).await,
            "PRE-FIX pool must be starved after 2 cancellations — no panic required"
        );
    }

    /// A pool reports its real state — what `/health` now publishes.
    #[tokio::test]
    async fn reports_real_capacity() {
        let pool = pool_of(3);
        assert_eq!(pool.capacity(), 3);
        assert_eq!(pool.available_capacity(), 3);
        assert_eq!(pool.lost(), 0);
    }
}
