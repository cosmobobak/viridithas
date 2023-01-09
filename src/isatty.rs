pub const STDOUT: u64 = 1;
pub const STDERR: u64 = 2;

pub static mut STDOUT_ISATTY: Option<bool> = None;
pub static mut STDERR_ISATTY: Option<bool> = None;

#[allow(unused_variables)]
pub fn isatty(fd: u64) -> bool {
    #[cfg(target_os = "linux")]
    unsafe {
        // Memoize to avoid making syscalls
        if fd == STDERR && STDERR_ISATTY.is_some() {
            return STDERR_ISATTY.unwrap();
        }
        if fd == STDOUT && STDOUT_ISATTY.is_some() {
            return STDOUT_ISATTY.unwrap();
        }
        let ret: i64;
        let ioctl: u64 = 16;
        let tcgets: u64 = 0x5401;
        let mut empty: [u64; 8] = [0; 8];
        std::arch::asm!(
          "syscall"
          , inout("rax") ioctl => ret
          ,    in("rdi") fd
          ,    in("rsi") tcgets
          ,    in("rdx") &mut empty
          ,   out("rcx") _
          ,   out("r11") _
        );
        let ret = ret == 0;
        if fd == STDERR {
            STDERR_ISATTY = Some(ret);
        }
        if fd == STDOUT {
            STDOUT_ISATTY = Some(ret);
        }
        ret
    }
    #[cfg(not(target_os = "linux"))]
    unsafe {
        if fd == STDERR && STDERR_ISATTY.is_some() {
            return STDERR_ISATTY.unwrap();
        }
        if fd == STDOUT && STDOUT_ISATTY.is_some() {
            return STDOUT_ISATTY.unwrap();
        }
        false
    }
}
