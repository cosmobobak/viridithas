use std::sync::atomic::AtomicUsize;
use std::sync::mpsc::{Receiver, SyncSender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::Scope;

// Handle for communicating with a worker thread.
// Contains a sender for sending messages to the worker thread,
// and a receiver for receiving messages from the worker thread.
pub struct WorkSender {
    // INVARIANT: Each send must be matched by a receive.
    sender: SyncSender<Box<dyn FnOnce() + Send>>,
    completion_signal: Arc<(Mutex<bool>, Condvar)>,
}

/// Handle for the receiver side of a worker thread.
struct WorkReceiver {
    receiver: Receiver<Box<dyn FnOnce() + Send>>,
    completion_signal: Arc<(Mutex<bool>, Condvar)>,
}

fn make_work_channel() -> (WorkSender, WorkReceiver) {
    let (sender, receiver) = std::sync::mpsc::sync_channel(0);
    let completion_signal = Arc::new((Mutex::new(false), Condvar::new()));

    (
        WorkSender {
            sender,
            completion_signal: Arc::clone(&completion_signal),
        },
        WorkReceiver {
            receiver,
            completion_signal,
        },
    )
}

pub struct ReceiverHandle<'scope> {
    completion_signal: &'scope Arc<(Mutex<bool>, Condvar)>,
    received: bool,
}

impl ReceiverHandle<'_> {
    pub fn join(mut self) {
        let (lock, cvar) = &**self.completion_signal;
        let mut completed = lock.lock().unwrap();
        while !*completed {
            completed = cvar.wait(completed).unwrap();
        }
        drop(completed);
        self.received = true;
    }
}

impl Drop for ReceiverHandle<'_> {
    fn drop(&mut self) {
        // When the receiver handle is dropped, we ensure that we have received something.
        assert!(
            self.received,
            "ReceiverHandle was dropped without receiving a value"
        );
    }
}

#[derive(Clone, Copy)]
pub struct ScopeExt<'a, 'scope, 'env> {
    scope: &'a Scope<'scope, 'env>,
    running: &'a Arc<AtomicUsize>,
}

impl<'scope, 'env> ScopeExt<'_, 'scope, 'env> {
    pub fn spawn_into<'comms, F>(self, f: F, thread: &'scope WorkerThread) -> ReceiverHandle<'scope>
    where
        F: FnOnce() + Send + 'scope,
    {
        // increase the running thread count
        let running = Arc::clone(self.running);

        running.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        // Safety: This file is structured such that threads never hold the data longer than is permissible.
        let f = unsafe {
            std::mem::transmute::<
                Box<dyn FnOnce() + Send + 'scope>,
                Box<dyn FnOnce() + Send + 'static>,
            >(Box::new(move || {
                // run the work
                f();
                // decrease the running thread count
                running.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            }))
        };

        // Reset the completion flag before sending the task
        {
            let (lock, _) = &*thread.comms.completion_signal;
            let mut completed = lock.lock().unwrap();
            *completed = false;
        }

        thread
            .comms
            .sender
            .send(f)
            .expect("Failed to send function to worker thread");

        ReceiverHandle {
            completion_signal: &thread.comms.completion_signal,
            // Important: We start with `received` as false.
            received: false,
        }
    }
}

fn make_worker_thread() -> WorkerThread {
    let (sender, receiver) = make_work_channel();

    let handle = std::thread::spawn(move || {
        while let Ok(work) = receiver.receiver.recv() {
            work();
            let (lock, cvar) = &*receiver.completion_signal;
            let mut completed = lock.lock().unwrap();
            *completed = true;
            drop(completed); // Release the lock before notifying
            cvar.notify_one();
        }
    });

    WorkerThread {
        handle,
        comms: sender,
    }
}

pub fn make_worker_threads(num_threads: usize) -> Vec<WorkerThread> {
    (0..num_threads).map(|_| make_worker_thread()).collect()
}

pub struct WorkerThread {
    handle: std::thread::JoinHandle<()>,
    comms: WorkSender,
}

impl WorkerThread {
    pub fn join(self) {
        drop(self.comms); // Drop the sender to signal the worker thread to finish
        self.handle.join().expect("Worker thread panicked");
    }
}

pub fn scope<'env, F>(f: F)
where
    F: for<'scope> FnOnce(ScopeExt<'_, 'scope, 'env>),
{
    std::thread::scope(|scope| {
        let running = Arc::new(AtomicUsize::new(0));
        let ext = ScopeExt {
            scope,
            running: &running,
        };
        f(ext);
        // wait for all threads to finish
        while running.load(std::sync::atomic::Ordering::SeqCst) > 0 {
            std::thread::yield_now();
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "ReceiverHandle was dropped without receiving a value")]
    fn test_work_sender_receiver() {
        let thread = make_worker_thread();

        scope(|s| {
            let _receiver_handle = s.spawn_into(
                || {
                    println!("Work is being done in the worker thread.");
                },
                &thread,
            );
        });

        thread.join();
    }

    #[test]
    fn test_work_sender_receiver_success() {
        let thread = make_worker_thread();

        scope(|s| {
            let receiver_handle = s.spawn_into(
                || {
                    println!("Work is being done in the worker thread.");
                },
                &thread,
            );
            receiver_handle.join(); // Ensure we receive the value
        });

        thread.join();
    }

    #[test]
    fn ub_example() {
        let mut global = String::new();
        let thread = make_worker_thread();
        {
            let local = String::from("test");
            scope(|s| {
                drop(s.spawn_into(
                    || {
                        std::thread::sleep(std::time::Duration::from_secs(1));
                        global = local.clone();
                    },
                    &thread,
                ));
            });
        }
        thread.join();

        assert_eq!(global, "test");
    }

    #[test]
    fn stdlib() {
        let mut global = String::new();
        {
            let local = String::from("test");
            std::thread::scope(|s| {
                std::mem::forget(s.spawn(|| {
                    std::thread::sleep(std::time::Duration::from_secs(1));
                    global = local.clone();
                }));
            });
        }

        assert_eq!(global, "test");
    }
}
