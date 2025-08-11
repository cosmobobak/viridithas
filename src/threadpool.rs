use std::thread::Scope;

fn spawn_through() {
    std::thread::scope(|s| {
        s.spawn(|| {});
    })
}

trait ScopeExt {
    fn spawn_into<F, T>(&'scope self, f: F) -> ()
    where
        F: FnOnce() -> T + Send + 'scope,
        T: Send + 'scope,
    {
    }
}

impl<'scope, 'env> ScopeExt for Scope<'scope, 'env> {
    fn spawn_into<F, T>(&'scope self, f: F) -> ScopedJoinHandle<'scope, T>
    where
        F: FnOnce() -> T + Send + 'scope,
        T: Send + 'scope,
    {
    }
}
