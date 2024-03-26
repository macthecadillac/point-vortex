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
    #[error(display = "Number of passive tracers not divisible by the number of threads")]
    NThreadsError,
    #[error(display = "Range start {} is greater than range end {}", _0, _1)]
    InvertedRange(f64, f64),
    #[error(display = "Empty range. Note that n must be greater than or equal to 2.")]
    EmptyRange,
}
