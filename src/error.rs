use err_derive::Error;

#[derive(Debug, Error)]
#[allow(dead_code)]
pub enum Error {
    #[error(display = "")]
    MissingConfig,
    #[error(display = "")]
    IOError(#[source] std::io::Error),
    #[error(display = "")]
    TOMLError(#[source] toml::de::Error),
}
