// SPDX-License-Identifier: AGPL-3.0-only

use std::{alloc::Layout, ops::Deref, ptr::NonNull};

use super::{RawCacheSet, threaded_memset_zero};
use crate::threadpool::WorkerThread;

#[derive(Debug)]
pub(super) struct Table {
    ptr: NonNull<RawCacheSet>,
    len: usize,
}

// Safety: Table uniquely owns its allocation.
unsafe impl Send for Table {}

// Safety: Shared access exposes only atomic entries.
unsafe impl Sync for Table {}

impl Table {
    pub const fn empty() -> Self {
        Self {
            ptr: NonNull::dangling(),
            len: 0,
        }
    }

    pub fn new(len: usize, threads: &[WorkerThread]) -> Self {
        if len == 0 {
            return Self::empty();
        }

        let layout = Layout::array::<RawCacheSet>(len).expect("Allocation is too large");

        let ptr = NonNull::new(platform::allocate(layout).cast())
            .unwrap_or_else(|| std::alloc::handle_alloc_error(layout));

        let table = Self { ptr, len };

        assert!(!threads.is_empty());

        // Safety: Zero initialize every atomic entry before any slice is exposed.
        unsafe { threaded_memset_zero(table.ptr.as_ptr().cast(), layout.size(), threads) };

        table
    }
}

impl Deref for Table {
    type Target = [RawCacheSet];

    fn deref(&self) -> &Self::Target {
        // Safety: We own the initialized entries or an aligned dangling empty slice
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for Table {
    fn drop(&mut self) {
        if self.len != 0 {
            let layout = Layout::array::<RawCacheSet>(self.len).unwrap();

            // Safety: This allocation belongs to us, and RawCacheSet needs no drop.
            unsafe { platform::deallocate(self.ptr.as_ptr().cast(), layout) };
        }
    }
}

#[cfg(target_os = "linux")]
mod platform {
    use std::{alloc::Layout, ptr, sync::OnceLock};

    fn page_sizes() -> (usize, usize) {
        static SIZES: OnceLock<(usize, usize)> = OnceLock::new();
        *SIZES.get_or_init(|| {
            // Safety: sysconf has no pointer arguments.
            let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };

            assert!(page > 0, "Cannot determine system page size");

            let page = page as usize;
            let huge =
                std::fs::read_to_string("/sys/kernel/mm/transparent_hugepage/hpage_pmd_size")
                    .ok()
                    .and_then(|s| s.trim().parse::<usize>().ok())
                    .filter(|&size| size >= page && size.is_power_of_two())
                    .unwrap_or(page);

            (page, huge)
        })
    }

    pub fn allocate(layout: Layout) -> *mut u8 {
        let (page, alignment) = page_sizes();

        assert!(layout.align() <= page);

        let Some(size) = layout.size().checked_next_multiple_of(page) else {
            return ptr::null_mut();
        };

        // Safety: All mappings are private, writable, and anonymous; trimming removes only the
        //         whole pages outside the retained allocation.
        unsafe {
            let map = |bytes| {
                libc::mmap(
                    ptr::null_mut(),
                    bytes,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };

            let mut memory = libc::MAP_FAILED;
            if let Some(reserved) = size.checked_add(alignment) {
                let mapping = map(reserved);
                if mapping != libc::MAP_FAILED {
                    let offset = (alignment - mapping.addr() % alignment) % alignment;
                    let aligned = mapping.cast::<u8>().add(offset);

                    if offset != 0 && libc::munmap(mapping, offset) != 0 {
                        std::process::abort();
                    }

                    if libc::munmap(aligned.add(size).cast(), alignment - offset) != 0 {
                        std::process::abort();
                    }

                    memory = aligned.cast();
                }
            }

            if memory == libc::MAP_FAILED {
                memory = map(size);
            }

            if memory == libc::MAP_FAILED {
                return ptr::null_mut();
            }

            libc::madvise(memory, size, libc::MADV_HUGEPAGE);

            memory.cast()
        }
    }

    pub unsafe fn deallocate(ptr: *mut u8, layout: Layout) {
        // Safety: The caller provides the original allocation here. munmap rounds the length to the
        // nearest page size.
        unsafe { libc::munmap(ptr.cast(), layout.size()) };
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

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
mod platform {
    use std::alloc::{self, Layout};

    pub fn allocate(layout: Layout) -> *mut u8 {
        // Safety: Non-empty and valid layouts are allocated.
        unsafe { alloc::alloc(layout) }
    }

    pub unsafe fn deallocate(ptr: *mut u8, layout: Layout) {
        // Safety: The caller provides the original allocation here.
        unsafe { alloc::dealloc(ptr, layout) };
    }
}
