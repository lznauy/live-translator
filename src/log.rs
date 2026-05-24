use std::sync::atomic::{AtomicBool, Ordering};

pub(crate) static QUIET: AtomicBool = AtomicBool::new(false);

pub fn set_quiet(q: bool) {
    QUIET.store(q, Ordering::Relaxed);
}

#[macro_export]
macro_rules! log {
    ($($arg:tt)*) => {
        if !$crate::log::QUIET.load(std::sync::atomic::Ordering::Relaxed) {
            eprintln!($($arg)*);
        }
    };
}
