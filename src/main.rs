use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_examples::hub_load_safetensors;
use candle_nn::{Module, VarBuilder};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::gemma2::{Config as GemmaConfig, Model as GemmaModel};
use dotenvy::dotenv;
use hf_hub::api::sync::ApiBuilder;
use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::Tokenizer;

// Custom transformer module to override the last layer
struct CustomGemmaModel {
    model: GemmaModel,
    device: Device,
}

impl CustomGemmaModel {
    pub fn new(vb: VarBuilder, config: GemmaConfig, device: Device) -> CandleResult<Self> {
        let mut model = GemmaModel::new(true, &config, vb)?;
        // Replace the last layer with a custom one
        let num_layers = config.num_hidden_layers;
        if let Some(last_layer) = model.layers.get_mut(num_layers - 1) {
            *last_layer = Arc::new(CustomLayer::new(Arc::clone(last_layer)));
        }
        Ok(Self { model, device })
    }

    pub fn generate(
        &self,
        input_ids: &Tensor,
        tokenizer: &Tokenizer,
        num_return_sequences: usize,
    ) -> Result<Vec<String>, Box<dyn Error + Send + Sync>> {
        let max_length = 2048;
        let eos_token_id = Some(tokenizer.token_to_id("<eos>").unwrap());
        let logits_processor = LogitsProcessor::new(1, Some(1.0), Some(0.95));
        // let test = LogitsProcessor::default()
        //     .with_temperature(1.0)
        //     .with_top_k(50)
        //     .with_top_p(0.95)
        //     .with_no_repeat_ngram_size(3);

        let mut generated_sequences = Vec::new();

        for _ in 0..num_return_sequences {
            let mut current_input = input_ids.clone();
            let mut generated_tokens = Vec::new();

            for _ in 0..max_length {
                // Forward pass: get logits from the model based on current input
                let logits = self.model.forward(&current_input, 0)?;

                // Process logits: apply temperature, top-k, top-p, etc.
                // let processed_logits = logits_processor.process(logits)?;

                // Sample the next token from the processed logits
                let next_token = logits_processor.sample(&logits)?;
                // let next_token = top_p_sampling(&processed_logits, 0.95)?;

                // Stop if EOS token is generated
                if Some(next_token) == eos_token_id {
                    break;
                }

                // Add the token to the generated sequence
                generated_tokens.push(next_token);

                // Prepare the next input (current tokens + new token)
                let new_tensor = Tensor::new(&[next_token], &self.device)?;
                current_input = Tensor::cat(&[&current_input, &new_tensor], 1)?;
            }

            // Decode the generated tokens into text using the tokenizer
            let text = tokenizer.decode(&generated_tokens, true)?;
            generated_sequences.push(text);
        }
        Ok(generated_sequences)
    }
}

// Define a custom layer that wraps the original layer
struct CustomLayer {
    layer: Arc<dyn Module>,
}

impl CustomLayer {
    pub fn new(original_layer: Arc<dyn Module>) -> Arc<Self> {
        Arc::new(Self {
            layer: original_layer,
        })
    }
}

impl Module for CustomLayer {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        // Custom logic before the forward pass
        println!("Custom logic before the forward pass");

        // Call the original forward method
        let output = self.layer.forward(xs)?;

        // Custom logic after the original forward pass
        println!("Custom logic after the forward pass");

        Ok(output)
    }
}

#[derive(Debug)]
struct ConfigFiles {
    pub config: PathBuf,
    pub tokenizer: PathBuf,
    pub weights: Vec<PathBuf>,
}

fn load_from_hub(model_id: &str, hf_token: &str) -> Result<ConfigFiles> {
    let hf_api_repo = ApiBuilder::new()
        .with_token(Some(hf_token.to_string()))
        .build()
        .unwrap()
        .model(model_id.to_string());
    Ok(ConfigFiles {
        config: hf_api_repo.get("config.json")?,
        tokenizer: hf_api_repo.get("tokenizer.json")?,
        weights: hub_load_safetensors(&hf_api_repo, "model.safetensors.index.json")?,
    })
}

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    dotenv().ok();
    // Set the device (CPU or GPU)
    let device: Device = Device::Cpu;
    // let device = if cuda_is_available() {
    //     Device::new_cuda(0)?
    // } else if metal_is_available() {
    //     Device::new_metal(0)?
    // } else {
    //     Device::Cpu
    // };
    let hf_token = env::var("HF_TOKEN").expect("HF_TOKEN not set in .env file");
    let config_files = load_from_hub("google/gemma-2-2b-it", &hf_token)?;

    let tokenizer = Tokenizer::from_file(config_files.tokenizer)?;
    let config: GemmaConfig = serde_json::from_slice(&std::fs::read(config_files.config)?)?;
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&config_files.weights, DType::F32, &device)? };

    let model = CustomGemmaModel::new(vb, config, device)?;

    // Define the invert_model function
    fn invert_model(
        tokenizer: &Tokenizer,
        model: &CustomGemmaModel,
        prompt: &str,
        num_return_sequences: usize,
    ) -> Result<String, Box<dyn Error + Send + Sync>> {
        let input_ids = tokenizer.encode(prompt, true)?.get_ids().to_vec();
        let input_ids = Tensor::new(input_ids, &model.device)?.unsqueeze(0)?;

        let outputs = model.generate(&input_ids, tokenizer, num_return_sequences)?;
        Ok(outputs[0].clone())
    }

    // Test the function with a sample prompt
    let prompt = "Tell me a joke about AI";
    let inverted_prompt = invert_model(&tokenizer, &model, prompt, 1)?;

    // Print the output
    println!("{}", inverted_prompt);

    Ok(())
}
