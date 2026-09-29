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
    use std::{alloc::Layout, mem::size_of, ptr, sync::Mutex};
    use windows_sys::Win32::{
        Foundation::{CloseHandle, ERROR_SUCCESS, GetLastError},
        Security::{
            AdjustTokenPrivileges, LookupPrivilegeValueA, SE_PRIVILEGE_ENABLED,
            TOKEN_ADJUST_PRIVILEGES, TOKEN_PRIVILEGES, TOKEN_QUERY,
        },
        System::{
            Memory::{
                GetLargePageMinimum, MEM_COMMIT, MEM_LARGE_PAGES, MEM_RELEASE, MEM_RESERVE,
                PAGE_READWRITE, VirtualAlloc, VirtualFree,
            },
            Threading::{GetCurrentProcess, OpenProcessToken},
        },
    };

    unsafe fn allocate_huge(bytes: usize) -> *mut u8 {
        // Safety: Referencing valid local structures and the token is closed after restoring the
        //         privilege's previous state.
        unsafe {
            let page = GetLargePageMinimum();
            if page == 0 || bytes < page {
                return ptr::null_mut();
            }

            let Some(size) = bytes.checked_next_multiple_of(page) else {
                return ptr::null_mut();
            };

            static LOCK: Mutex<()> = Mutex::new(());
            let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());

            let mut token = ptr::null_mut();
            if OpenProcessToken(
                GetCurrentProcess(),
                TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY,
                &mut token,
            ) == 0
            {
                return ptr::null_mut();
            }

            let mut requested: TOKEN_PRIVILEGES = std::mem::zeroed();
            requested.PrivilegeCount = 1;
            requested.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;

            if LookupPrivilegeValueA(
                ptr::null(),
                c"SeLockMemoryPrivilege".as_ptr().cast(),
                &mut requested.Privileges[0].Luid,
            ) == 0
            {
                CloseHandle(token);
                return ptr::null_mut();
            }

            let mut previous: TOKEN_PRIVILEGES = std::mem::zeroed();
            let mut previous_size = size_of::<TOKEN_PRIVILEGES>() as u32;

            let adjusted = AdjustTokenPrivileges(
                token,
                0,
                &requested,
                previous_size,
                &mut previous,
                &mut previous_size,
            );
            let error = GetLastError();

            let memory = if adjusted != 0 && error == ERROR_SUCCESS {
                VirtualAlloc(
                    ptr::null(),
                    size,
                    MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                    PAGE_READWRITE,
                )
            } else {
                ptr::null_mut()
            };

            if adjusted != 0 && previous.PrivilegeCount != 0 {
                AdjustTokenPrivileges(token, 0, &previous, 0, ptr::null_mut(), ptr::null_mut());
            }

            CloseHandle(token);

            memory.cast()
        }
    }

    pub fn allocate(layout: Layout) -> *mut u8 {
        // Safety: Either way, private writable memory with sufficient alignment (for RawCacheSet)
        //         is allocated; VirtualFree will release either kind.
        unsafe {
            let memory = allocate_huge(layout.size());

            if memory.is_null() {
                VirtualAlloc(
                    ptr::null(),
                    layout.size(),
                    MEM_RESERVE | MEM_COMMIT,
                    PAGE_READWRITE,
                )
                .cast()
            } else {
                memory
            }
        }
    }

    pub unsafe fn deallocate(ptr: *mut u8, _: Layout) {
        // Safety: The caller provides the original allocation here.
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
