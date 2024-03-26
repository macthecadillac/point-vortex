use err_derive::Error;

#[derive(Debug, Error)]
#[allow(dead_code)]
pub(crate) enum Error {
    #[error(display = "")]
    MissingConfig,
    #[error(display = "")]
    IOError(#[source] std::io::Error),
    #[error(display = "")]
    TOMLError(#[source] toml::de::Error),
    #[error(display = "number of passive tracers not divisible by the number of threads")]
    NThreadsError,
    #[error(display = "")]
    TryForEachBreak
}
