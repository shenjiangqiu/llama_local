use tracing_subscriber::{filter::LevelFilter, EnvFilter};

pub fn init_logger() {
    tracing_subscriber::fmt::SubscriberBuilder::default()
        .with_env_filter(
            EnvFilter::builder()
                .with_default_directive(LevelFilter::INFO.into())
                .from_env_lossy(),
        )
        .try_init()
        .ok();
}
