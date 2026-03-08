#[expect(warnings)]
mod bindings;

use std::sync::LazyLock;

pub struct Numa(private::Token);

mod private {
    /// A token type to prevent external construction of `Numa`.
    pub struct Token;
}

/// Wrapper to make `*mut bitmask` usable in statics.
pub struct BitmaskPtr(*mut bindings::bitmask);
// SAFETY: The pointer is only used in a read-only fashion after being initialised, and the underlying bitmask is never deallocated.
unsafe impl Send for BitmaskPtr {}
// SAFETY: The pointer is only used in a read-only fashion after being initialised, and the underlying bitmask is never deallocated.
unsafe impl Sync for BitmaskPtr {}

/// Token for `libnuma` initialisation.
/// Is only `Some` if `libnuma` is available.
pub static NUMA: LazyLock<Option<Numa>> = LazyLock::new(|| {
    if !Numa::available() {
        println!("info string NUMA is not available on this system");
        println!("info string this is likely HIGHLY UNDESIRABLE and may indicate a bug");
        return None;
    }

    let numa = Numa(private::Token);

    println!("info string NUMA is available on this system");
    println!(
        "info string NUMA is available with {} configured nodes.",
        numa.node_count()
    );

    Some(numa)
});

#[expect(clippy::undocumented_unsafe_blocks, clippy::unused_self)]
impl Numa {
    pub fn available() -> bool {
        // Before any other calls in libnuma can be used, numa_available() must be called.
        // If it returns -1, all other functions in the library are undefined.
        unsafe { bindings::numa_available() != -1 }
    }

    /// A mapping of NUMA nodes to CPU bitmasks, used for thread binding.
    pub fn thread_mapping(&self) -> &'static [BitmaskPtr] {
        // The thread mapping is stored in a static, s.t. we can keep the libnuma
        // fiddling to an absolute minimum. It also means that we only leak a constant
        // quantity of memory for these bitmask types.
        static THREAD_MAPPING: LazyLock<Vec<BitmaskPtr>> = LazyLock::new(|| {
            let make_bitmask = |node: i32| -> BitmaskPtr {
                // > numa_allocate_cpumask() returns a bitmask of a size equal to
                // > the kernel's cpu mask (kernel type cpumask_t). In other words,
                // > large enough to represent NR_CPUS cpus. This number of cpus
                // > can be gotten by calling numa_num_possible_cpus().
                // > The bitmask is zero-filled.
                // I take this to mean that if we have, say, 128 possible CPUs,
                // we get a bitmask with 128 bits (16 bytes) of storage.
                let cpumask = unsafe { bindings::numa_allocate_cpumask() };
                // > numa_node_to_cpus() converts a node number to a bitmask of CPUs.
                // > The user must pass a bitmask structure with a mask buffer long
                // > enough to represent all possible cpu's. [sic]
                // > Use numa_allocate_cpumask() to create it. If the bitmask is not
                // > long enough errno will be set to ERANGE and -1 returned.
                // > On success 0 is returned.
                let res = unsafe { bindings::numa_node_to_cpus(node, cpumask) };
                assert_eq!(res, 0, "failed to get CPU mask for NUMA node {node}");
                // box it up!
                BitmaskPtr(cpumask)
            };

            // Get the highest node number available on the current system.
            let max_node = unsafe { bindings::numa_max_node() };

            // Get the CPU bitmask for each node, and store it in a vector.
            (0..=max_node).map(make_bitmask).collect()
        });

        &THREAD_MAPPING
    }

    /// The number of NUMA nodes in the system.
    pub fn node_count(&self) -> usize {
        self.thread_mapping().len()
    }

    /// Get the NUMA node for a given thread ID.
    pub fn get_node(&self, id: usize) -> usize {
        id % self.node_count()
    }

    /// Bind the current thread to the NUMA node corresponding to the given thread ID.
    pub fn bind_thread(&self, id: usize) {
        let map = self.thread_mapping();
        let node = self.get_node(id);
        let mask = &map[node];
        unsafe {
            // > numa_run_on_node_mask() runs the current task and its children
            // > only on nodes specified in nodemask. They will not migrate to
            // > CPUs of other nodes until the node affinity is reset with a new
            // > call to numa_run_on_node_mask() or numa_run_on_node(). Passing
            // > numa_all_nodes permits the kernel to schedule on all nodes again.
            // > On success, 0 is returned; on error -1 is returned, and errno is
            // > set to indicate the error.
            bindings::numa_run_on_node_mask(mask.0);
        }
    }
}

/// Smart pointer that replicates read-only data across NUMA nodes.
/// The data is allocated once per NUMA node.
pub struct NumaReplicated<T> {
    /// The data for each NUMA node. The length of this vector should be equal to the number of NUMA nodes in the system.
    data: Vec<*mut T>,
    /// Whether the memory was allocated with NUMA support.
    /// If `false`, the data is stored in a single copy and the vector has length 1.
    numa_allocated: bool,
    _marker: std::marker::PhantomData<T>,
}

// SAFETY: NumaReplicated is just a normal smart pointer
// from a safety perspective.
unsafe impl<T> Send for NumaReplicated<T> {}
// SAFETY: NumaReplicated is just a normal smart pointer
// from a safety perspective.
unsafe impl<T> Sync for NumaReplicated<T> {}

impl<T> Drop for NumaReplicated<T> {
    fn drop(&mut self) {
        if self.numa_allocated {
            for &ptr in &self.data {
                // SAFETY: If `numa_allocated` is true, the data was allocated with
                // `numa_alloc_onnode`, and we must free each copy with `numa_free`.
                unsafe {
                    // > numa_free() frees memory allocated by numa_alloc_onnode() or numa_alloc_interleaved().
                    // > The size argument must be the same as the one passed to the allocation function.
                    bindings::numa_free(ptr.cast::<std::ffi::c_void>(), std::mem::size_of::<T>());
                }
            }
        } else {
            // SAFETY: If `numa_allocated` is false, we have a single copy of the data that
            // was allocated with `Box::into_raw`, and we can safely deallocate it here.
            unsafe {
                drop(Box::from_raw(self.data[0]));
            }
        }
    }
}

/// Equivalent to Copy, without risking a user making some huge type copyable.
///
/// # Safety
///
/// The implementor of this trait must be safely copyable via memcpy.
pub unsafe trait NumaReplicable {}

impl<T: NumaReplicable> NumaReplicated<T> {
    /// Create a new `NumaReplicated` by copying the given data to each NUMA node.
    pub fn new(data: &T) -> Self {
        let Some(numa) = &*NUMA else {
            // If NUMA is not available, just store a single copy of the data.
            // We may be working with very large data here, so we wish to avoid
            // placing a `T` on the stack.
            let ptr = Box::leak(Box::<T>::new_uninit()).as_mut_ptr();
            // SAFETY: `T` is `Copy`, and so it can be duplicated simply by copying bits.
            unsafe {
                std::ptr::copy_nonoverlapping::<T>(data, ptr, 1);
            }
            return Self {
                data: vec![ptr],
                numa_allocated: false,
                _marker: std::marker::PhantomData,
            };
        };

        let nodes = numa.node_count();

        let memory: Vec<*mut T> = (0..nodes)
            .map(|node| {
                // > numa_alloc_onnode() allocates memory on a specific node.
                // > The size argument will be rounded up to a multiple of the system page size.
                // > If the specified node is externally denied to this process, this call will fail.
                // > This function is relatively slow compared to the malloc(3) family of functions.
                // > The memory must be freed with numa_free(). On errors NULL is returned.

                // SAFETY: Nothing particularly scary seems to be going on here, and we panic
                // on failure, so this should be fine.
                let ptr = unsafe {
                    #[expect(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
                    bindings::numa_alloc_onnode(std::mem::size_of::<T>(), node as i32)
                }
                .cast::<T>();

                assert!(!ptr.is_null(), "Failed to allocate memory on NUMA node");

                // SAFETY: `T` is `Copy`, and so it can be duplicated simply by copying bits.
                unsafe {
                    std::ptr::copy_nonoverlapping::<T>(data, ptr, 1);
                }

                ptr
            })
            .collect();

        Self {
            data: memory,
            numa_allocated: true,
            _marker: std::marker::PhantomData,
        }
    }

    /// Get a reference to the data for the current thread's NUMA node.
    pub fn get(&self, id: usize) -> &T {
        let Some(numa) = &*NUMA else {
            // If NUMA is not available, just return the single copy of the data.
            // SAFETY: If NUMA is not available, we have a single copy of the data that
            // was allocated with `Box::into_raw`, and we can safely return a reference to it here.
            return unsafe { &*self.data[0] };
        };

        let node = numa.get_node(id);
        // SAFETY: The pointer at `self.data[node]` is valid for reads,
        // and the data it points to is properly initialised.
        unsafe { &*self.data[node] }
    }
}
