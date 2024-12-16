use std::{thread, time::Duration};

use kalosm::language::*;
use llama_local::init_logger;
use tracing::info;

#[tokio::main]
async fn main() {
    init_logger();
    let model = Llama::new_chat().await.unwrap();
    let mut chat = Chat::builder(model.clone())
        .with_system_prompt("You are a chatbot named Cloth.")
        .build();

    chat.add_message("Hello, what's your name? just type your name.")
        .to_std_out()
        .await
        .unwrap();
    chat.add_message("What's 1+1=?")
        .to_std_out()
        .await
        .unwrap();
    model
        .run_sync(|x| {
            info!("Printing cache");
            Box::pin(async move {
                let kv_cache = x.get_kv_cache();
                kv_cache.print();
            })
        })
        .unwrap();

    thread::sleep(Duration::from_secs(3));
}
