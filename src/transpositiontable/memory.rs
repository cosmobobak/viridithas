// SPDX-License-Identifier: AGPL-3.0-only

use std::{alloc::Layout, ops::Deref, ptr::NonNull};

use super::{RawCacheSet, threaded_memset_zero};
use crate::threadpool::WorkerThread;

#[derive(Debug)]
pub(super) struct Table {
    ptr: NonNull<[RawCacheSet]>,
}

// SAFETY: Table uniquely owns its allocation.
unsafe impl Send for Table {}

// SAFETY: Shared access exposes only atomic entries.
unsafe impl Sync for Table {}

impl Table {
    pub const fn empty() -> Self {
        Self {
            ptr: NonNull::slice_from_raw_parts(NonNull::dangling(), 0),
        }
    }

    pub fn new(len: usize, threads: &[WorkerThread]) -> Self {
        if len == 0 {
            return Self::empty();
        }

        let layout = Layout::array::<RawCacheSet>(len).expect("Allocation is too large");

        let ptr = NonNull::new(platform::allocate(layout).cast())
            .unwrap_or_else(|| std::alloc::handle_alloc_error(layout));

        let table = Self {
            ptr: NonNull::slice_from_raw_parts(ptr, len),
        };

        assert!(!threads.is_empty());

        // SAFETY: Zeroed memory is a legal bitpattern for AtomicUXX.
        unsafe { threaded_memset_zero(table.ptr.as_ptr().cast(), layout.size(), threads) };

        table
    }
}

impl Deref for Table {
    type Target = [RawCacheSet];

    fn deref(&self) -> &Self::Target {
        // SAFETY: Slice is empty or an allocation owned by Table.
        unsafe { self.ptr.as_ref() }
    }
}

impl Drop for Table {
    fn drop(&mut self) {
        if !self.ptr.is_empty() {
            let layout = Layout::array::<RawCacheSet>(self.ptr.len()).unwrap();

            // SAFETY: RawCacheSet is POD and allocation is owned by Table.
            unsafe { platform::deallocate(self.ptr.as_ptr().cast(), layout) };
        }
    }
}

#[cfg(target_os = "windows")]
mod platform {
    use std::{alloc::Layout, ptr, sync::OnceLock};
    use windows_sys::Win32::{
        Foundation::{CloseHandle, ERROR_SUCCESS, GetLastError, LUID},
        Security::{
            AdjustTokenPrivileges, LUID_AND_ATTRIBUTES, LookupPrivilegeValueA,
            SE_PRIVILEGE_ENABLED, TOKEN_ADJUST_PRIVILEGES, TOKEN_PRIVILEGES,
        },
        System::{
            Memory::{
                GetLargePageMinimum, MEM_COMMIT, MEM_LARGE_PAGES, MEM_RELEASE, MEM_RESERVE,
                PAGE_READWRITE, VIRTUAL_ALLOCATION_TYPE, VirtualAlloc, VirtualFree,
            },
            Threading::{GetCurrentProcess, OpenProcessToken},
        },
    };

    fn large_page_size() -> Option<usize> {
        static SIZE: OnceLock<Option<usize>> = OnceLock::new();
        *SIZE.get_or_init(|| {
            // SAFETY: GetLargePageMinimum has no preconditions.
            let size = unsafe { GetLargePageMinimum() };
            (size != 0 && enable_lock_memory_privilege()).then_some(size)
        })
    }

    fn enable_lock_memory_privilege() -> bool {
        // SAFETY: All pointers are to valid locals, and the token is closed before returning.
        unsafe {
            let mut token = ptr::null_mut();
            if OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES, &raw mut token) == 0 {
                return false;
            }

            let mut privileges = TOKEN_PRIVILEGES {
                PrivilegeCount: 1,
                Privileges: [LUID_AND_ATTRIBUTES {
                    Luid: LUID::default(),
                    Attributes: SE_PRIVILEGE_ENABLED,
                }],
            };

            let name = c"SeLockMemoryPrivilege".as_ptr().cast();
            let luid = &raw mut privileges.Privileges[0].Luid;

            let enabled = LookupPrivilegeValueA(ptr::null(), name, luid) != 0
                && AdjustTokenPrivileges(
                    token,
                    0,
                    &raw const privileges,
                    0,
                    ptr::null_mut(),
                    ptr::null_mut(),
                ) != 0
                && GetLastError() == ERROR_SUCCESS;

            CloseHandle(token);
            enabled
        }
    }

    fn virtual_alloc(size: usize, flags: VIRTUAL_ALLOCATION_TYPE) -> *mut u8 {
        // SAFETY: Allocating fresh memory.
        unsafe {
            VirtualAlloc(
                ptr::null(),
                size,
                MEM_RESERVE | MEM_COMMIT | flags,
                PAGE_READWRITE,
            )
            .cast()
        }
    }

    pub fn allocate(layout: Layout) -> *mut u8 {
        large_page_size()
            .map(|page| virtual_alloc(layout.size().next_multiple_of(page), MEM_LARGE_PAGES))
            .filter(|ptr| !ptr.is_null())
            .unwrap_or_else(|| virtual_alloc(layout.size(), 0))
    }

    pub unsafe fn deallocate(ptr: *mut u8, _: Layout) {
        // SAFETY: `ptr` came from VirtualAlloc.
        unsafe { VirtualFree(ptr.cast(), 0, MEM_RELEASE) };
    }
}

#[cfg(not(target_os = "windows"))]
mod platform {
    use std::alloc::{self, Layout};

    use crate::util::MEGABYTE;

    const HUGE_PAGE: usize = if cfg!(target_os = "linux") {
        2 * MEGABYTE
    } else {
        1
    };

    fn huge_layout(layout: Layout) -> Layout {
        layout.align_to(HUGE_PAGE).unwrap().pad_to_align()
    }

    pub fn allocate(layout: Layout) -> *mut u8 {
        let layout = huge_layout(layout);

        // SAFETY: `layout` has a non-zero size.
        let ptr = unsafe { alloc::alloc(layout) };

        #[cfg(target_os = "linux")]
        if !ptr.is_null() {
            // SAFETY: Range correct.
            unsafe { libc::madvise(ptr.cast(), layout.size(), libc::MADV_HUGEPAGE) };
        }

        ptr
    }

    pub unsafe fn deallocate(ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` and `layout` must be the same as in the original allocation.
        unsafe { alloc::dealloc(ptr, huge_layout(layout)) };
    }
}
