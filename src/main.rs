use std::io::{self, IsTerminal, Read};
use std::time::Instant;

use argh::FromArgs;

/// QORA text-only CPU inference engine
#[derive(FromArgs)]
struct Args {
    /// prompt text (takes precedence over stdin when both are supplied)
    #[argh(option)]
    prompt: Option<String>,

    /// maximum number of generated tokens
    #[argh(option)]
    max_tokens: Option<usize>,

    /// thinking token budget
    #[argh(option)]
    think_budget: Option<usize>,

    /// disable thinking
    #[argh(switch)]
    no_think: bool,

    /// use greedy decoding (temp. = 0)
    #[argh(switch)]
    greedy: bool,

    /// display thinking tokens
    #[argh(switch)]
    show_think: bool,

    /// print diagnostic and progress information to stderr
    #[argh(switch)]
    verbose: bool,

    /// print version information
    #[argh(switch, short = 'V')]
    version: bool,
}

fn read_prompt(explicit_prompt: Option<String>) -> io::Result<String> {
    if let Some(prompt) = explicit_prompt {
        return Ok(prompt);
    }

    let stdin = io::stdin();
    if !stdin.is_terminal() {
        let mut prompt = String::new();
        stdin.lock().read_to_string(&mut prompt)?;

        if prompt.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "stdin was redirected or piped, but contained no prompt text",
            ));
        }

        return Ok(prompt);
    }

    Ok(String::from("Hello, how are you?"))
}

fn main() {
    let args: Args = argh::from_env();

    if args.version {
        println!("{} {}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"));
        return;
    }

    let prompt = match read_prompt(args.prompt) {
        Ok(prompt) => prompt,
        Err(err) => {
            eprintln!("Error reading prompt: {err}");
            std::process::exit(2);
        }
    };

    let no_think = args.no_think;
    let greedy = args.greedy;
    let show_think = args.show_think;
    let verbose = args.verbose;

    // System awareness
    let sys = qora::system::SystemInfo::detect();
    let limits = sys.smart_limits();
    if verbose {
        eprintln!("QORA - Pure Rust Text-Only CPU Inference Engine");
        eprintln!(
            "System: {} MB RAM ({} MB free), {} threads",
            sys.total_ram_mb, sys.available_ram_mb, sys.cpu_threads
        );
    }

    // Preserve the existing smart defaults while allowing explicit overrides
    let mut max_tokens = args.max_tokens.unwrap_or(limits.default_max_tokens);
    let mut think_budget = args.think_budget.unwrap_or(limits.default_think_budget);

    // Hard cap: even explicit values get clamped on weak systems
    if think_budget > limits.max_think_budget {
        if verbose {
            eprintln!(
                "System cap: think-budget {} → {}",
                think_budget, limits.max_think_budget
            );
        }
        think_budget = limits.max_think_budget;
    }
    if max_tokens > limits.max_tokens {
        if verbose {
            eprintln!(
                "System cap: max-tokens {} → {}",
                max_tokens, limits.max_tokens
            );
        }
        max_tokens = limits.max_tokens;
    }

    if verbose {
        if let Some(msg) = limits.warning {
            eprintln!("WARNING: {msg}");
        }
    }

    // Auto-adjust: ensure enough room for the answer after thinking
    if !no_think {
        let min_answer = 256;
        if max_tokens < think_budget + min_answer {
            max_tokens = think_budget + min_answer;
            // Re-apply hard cap
            if max_tokens > limits.max_tokens {
                max_tokens = limits.max_tokens;
            }
            if verbose {
                eprintln!(
                    "Auto-adjusted max-tokens to {} (think-budget {} + {} answer room)",
                    max_tokens, think_budget, min_answer
                );
            }
        }
    }

    let mode_str = if no_think { "no-think" } else { "think" };
    if verbose {
        eprintln!("Prompt: {prompt}");
        eprintln!("Mode: {mode_str}");
    }

    // Load model from same directory as the executable
    let exe_dir = std::env::current_exe()
        .expect("Cannot determine executable path")
        .parent()
        .unwrap()
        .to_path_buf();
    let model_path = exe_dir.join("model.qor4b");
    if verbose {
        eprintln!("Loading model from {}...", model_path.display());
    }
    let t0 = Instant::now();
    let weights = qora::save::load_model_text_with_verbosity(&model_path, verbose)
        .expect("Failed to load model");
    let mem_mb = weights.memory_bytes() / (1024 * 1024);
    if verbose {
        eprintln!(
            "Model loaded in {:.1?} ({} format, {mem_mb} MB)",
            t0.elapsed(),
            weights.format_name
        );
    }

    // Load tokenizer from same directory as the executable
    let tokenizer_path = exe_dir.join("tokenizer.json");
    let tokenizer = qora::tokenizer::QoraTokenizer::from_file(&tokenizer_path)
        .expect("Failed to load tokenizer");

    let temperature = if greedy {
        0.0
    } else if no_think {
        0.7
    } else {
        1.0
    };

    let params = qora::generate::GenerateParams {
        max_new_tokens: max_tokens,
        eos_token_id: weights.config.eos_token_id,
        temperature,
        top_p: 0.95,
        top_k: 20,
        think: !no_think,
        show_think,
        repetition_penalty: 1.0,
        presence_penalty: 1.5,
        think_budget,
    };

    // Text-only CPU inference
    qora::generate::generate_with_verbosity(&weights, &tokenizer, &prompt, &params, verbose);
}
