use std::sync::mpsc::{Receiver, Sender};
use std::thread::Scope;

// Handle for communicating with a worker thread.
// Contains a sender for sending messages to the worker thread,
// and a receiver for receiving messages from the worker thread.
pub struct WorkSender {
    // INVARIANT: Each send must be matched by a receive.
    sender: Sender<Box<dyn FnOnce() + Send>>,
    receiver: Receiver<()>,
}

/// Handle for the receiver side of a worker thread.
struct WorkReceiver {
    receiver: Receiver<Box<dyn FnOnce() + Send>>,
    sender: Sender<()>,
}

fn make_work_channel() -> (WorkSender, WorkReceiver) {
    let (sender, receiver) = std::sync::mpsc::channel();
    let (result_sender, result_receiver) = std::sync::mpsc::channel();

    (
        WorkSender {
            sender,
            receiver: result_receiver,
        },
        WorkReceiver {
            receiver,
            sender: result_sender,
        },
    )
}

pub struct ReceiverHandle<'scope> {
    receiver: &'scope Receiver<()>,
    received: bool,
}

impl ReceiverHandle<'_> {
    // Receives a value from the worker thread.
    // This will block until a value is available.
    pub fn receive(mut self) {
        self.receiver
            .recv()
            .expect("Failed to receive value from worker thread");
        self.received = true; // Mark that we have received a value
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

        thread
            .comms
            .sender
            .send(f)
            .expect("Failed to send function to worker thread");

        ReceiverHandle {
            receiver: &thread.comms.receiver,
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
            receiver.sender.send(()).expect("Failed to send result");
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
            receiver_handle.receive(); // Ensure we receive the value
        });

        thread.join();
    }
}
