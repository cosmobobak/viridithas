use std::sync::mpsc::{Receiver, SyncSender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::Scope;

use vec1::Vec1;

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

pub trait ScopeExt<'scope, 'env> {
    fn spawn_into<F>(&'scope self, f: F, comms: &'scope WorkerThread) -> ReceiverHandle<'scope>
    where
        F: FnOnce() + Send + 'scope;
}

impl<'scope, 'env> ScopeExt<'scope, 'env> for Scope<'scope, 'env> {
    fn spawn_into<'comms, F>(
        &'scope self,
        f: F,
        thread: &'scope WorkerThread,
    ) -> ReceiverHandle<'scope>
    where
        F: FnOnce() + Send + 'scope,
    {
        // Safety: This file is structured such that threads never hold the data longer than is permissible.
        let f = unsafe {
            std::mem::transmute::<
                Box<dyn FnOnce() + Send + 'scope>,
                Box<dyn FnOnce() + Send + 'static>,
            >(Box::new(f))
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

/// Create some number of worker threads. Panics if `num_threads` is zero.
pub fn make_worker_threads(num_threads: usize) -> Vec1<WorkerThread> {
    (0..num_threads)
        .map(|_| make_worker_thread())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "ReceiverHandle was dropped without receiving a value")]
    fn test_work_sender_receiver() {
        let thread = make_worker_thread();

        std::thread::scope(|s| {
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

        std::thread::scope(|s| {
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
}
