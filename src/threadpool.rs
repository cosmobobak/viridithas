use std::{
    sync::mpsc::{self, Receiver, Sender},
    thread::{self, JoinHandle},
};

pub trait Task: Send {
    fn execute(&mut self);
}

struct Search;

impl Task for Search {
    fn execute(&mut self) {
        todo!()
    }
}

struct InitThreadData {
    message: String,
}

impl Task for InitThreadData {
    fn execute(&mut self) {
        // Placeholder for thread initialization
        // Real initialization would happen outside the threadpool
        println!("Initializing thread: {}", self.message);
    }
}

enum Message {
    Task(Box<dyn Task>),
    Quit,
}

unsafe impl Send for Message {}

struct Worker {
    /// The handle associated with this worker's thread.
    handle: JoinHandle<()>,
    /// Put stuff in here to give it to the thread.
    sender: Sender<Message>,
    /// This pings every time the thread finishes a task.
    bell: Receiver<()>,
}

impl Worker {
    fn new() -> Self {
        let (task_sender, task_receiver) = mpsc::channel::<Message>();
        let (bell_sender, bell_receiver) = mpsc::channel::<()>();

        let handle = thread::spawn(move || {
            while let Ok(message) = task_receiver.recv() {
                match message {
                    Message::Task(mut task) => {
                        task.execute();
                        let _ = bell_sender.send(());
                    }
                    Message::Quit => break,
                }
            }
        });

        Self {
            handle,
            sender: task_sender,
            bell: bell_receiver,
        }
    }

    fn send_task(&self, task: Box<dyn Task>) -> Result<(), Box<dyn Task>> {
        self.sender
            .send(Message::Task(task))
            .map_err(|e| match e.0 {
                Message::Task(task) => task,
                Message::Quit => unreachable!(),
            })
    }

    fn quit(self) -> JoinHandle<()> {
        let _ = self.sender.send(Message::Quit);
        self.handle
    }
}

pub struct ThreadPool {
    workers: Vec<Worker>,
}

impl ThreadPool {
    pub fn new() -> Self {
        Self {
            workers: Vec::new(),
        }
    }

    pub fn resize_to(&mut self, num_threads: usize) {
        let current_len = self.workers.len();

        if num_threads > current_len {
            // Add more workers
            for _ in current_len..num_threads {
                self.workers.push(Worker::new());
            }
        } else if num_threads < current_len {
            // Remove workers
            let workers_to_remove = self.workers.split_off(num_threads);
            for worker in workers_to_remove {
                let _ = worker.quit().join();
            }
        }
    }

    pub fn submit_task(&self, task: Box<dyn Task>) -> Result<(), Box<dyn Task>> {
        if self.workers.is_empty() {
            return Err(task);
        }

        // Simple round-robin distribution for now
        // In a real implementation, you might want to choose the least busy worker
        let worker_idx = 0; // For simplicity, always use first worker
        self.workers[worker_idx].send_task(task)
    }

    pub fn wait_for_task_completion(&self, worker_idx: usize) -> Result<(), ()> {
        if worker_idx >= self.workers.len() {
            return Err(());
        }

        // Wait for the bell to ring indicating task completion
        self.workers[worker_idx].bell.recv().map_err(|_| ())
    }

    pub fn num_workers(&self) -> usize {
        self.workers.len()
    }

    pub fn shutdown(self) {
        for worker in self.workers {
            let _ = worker.quit().join();
        }
    }
}
