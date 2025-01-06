# ParrotFlume

ParrotFlume is a versatile command-line tool designed to interact with OpenAI-compatible APIs, providing a seamless interface for chat, file transformation, and task execution using large language models (LLMs). The name "ParrotFlume" is inspired by the concept of "flume" referring to input/output pipes and "parrot" symbolizing the stochastic nature of LLMs, often referred to as "stochastic parrots."

## Features

- **Interactive Chat Mode**: Engage in a conversational interface with the LLM, supporting multiline input, chat history management, and more.
- **One-Shot Mode**: Provide a single prompt and receive an immediate response from the LLM.
- **File Transformation**: Transform file content based on a given prompt, ideal for batch processing or automated tasks.
- **Task Execution**: Perform specific tasks on file content using the LLM, with the ability to customize the prompt.
- **Function Calling**: Utilize built-in functions for mathematical operations, date retrieval, and more.
- **Markdown, LaTeX, and Color Support**: Enhanced output formatting with Markdown, LaTeX, and ANSI color codes.

## Installation

To install ParrotFlume, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/ParrotFlume.git
cd ParrotFlume
pip install -r requirements.txt
```

## Configuration

ParrotFlume uses a TOML configuration file to manage API providers, model settings, and global options. The configuration file should be placed in the appropriate user configuration directory.

### Example Configuration (`parrotflume.config.toml`)

```toml
[global_options]
temperature = 0.1  # Default temperature for the model
max_tokens = 4096  # Maximum number of tokens to generate
markdown = true    # Enable markdown rendering
color = true       # Enable colored output
latex = true       # Enable LaTeX replacement
func = true        # Enable function calling

# API providers
[[api_providers]]
name = "openai"
base_url = "https://api.openai.com/v1/"
api_key = "<yourapikeyhere>"
model = "gpt-4o"

[[api_providers]]
name = "deepseek"
base_url = "https://api.deepseek.com/v1/"
api_key = "<yourapikeyhere>"
model = "deepseek-chat"

[[api_providers]]
name = "llama.cpp"
base_url = "http://localhost:8080/v1/"
api_key = "sk-no-key-required"  # not used, NOT allowed to be empty for llama.cpp
model = ""   # not used, allowed to be empty for llama.cpp
```

### Configuration File Location (Linux)

On Linux, the configuration file is typically located in the user's configuration directory:

```bash
~/.config/parrotflume/parrotflume.config.toml
```

You can create the directory and file manually if it doesn't exist:

```bash
mkdir -p ~/.config/parrotflume
touch ~/.config/parrotflume/parrotflume.config.toml
```

## Usage

### Command-Line Parameters

#### General Parameters
- **`--chat`**: Start an interactive chat session with the LLM.
- **`--oneshot "<prompt>"`**: Provide a single prompt and get an immediate response. Example:
  ```bash
  python -m parrotflume --oneshot "Explain the concept of quantum entanglement in simple terms."
  ```
- **`--transform "<prompt>" [filename]`**: Transform the content of a file using a prompt. If no filename is provided, reads from `stdin`. Example:
  ```bash
  python -m parrotflume --transform "Translate this text to French" input.txt
  ```
  Using `stdin` and `stdout`:
  ```bash
  cat input.txt | python -m parrotflume --transform "Summarize this text" > output.txt
  ```
- **`--perform "<prompt>" [filename]`**: Perform a task on the content of a file. If no filename is provided, reads from `stdin`. Example:
  ```bash
  python -m parrotflume --perform "Extract all dates from this text" input.txt
  ```
  Using `stdin` and `stdout`:
  ```bash
  cat input.txt | python -m parrotflume --perform "Extract all email addresses" > emails.txt
  ```
- **`--list`**: List all available models from the configured API provider. Example:
  ```bash
  python -m parrotflume --list
  ```

#### API Configuration Parameters
- **`--api-provider <provider>`**: Set the API provider (e.g., `openai`, `deepseek`, `llama.cpp`). Example:
  ```bash
  python -m parrotflume --api-provider openai --chat
  ```
- **`--base-url <url>`**: Set the base URL for the API provider. Example:
  ```bash
  python -m parrotflume --base-url "https://api.openai.com/v1/" --chat
  ```
- **`--key <key>`**: Set the API key for the API provider. Example:
  ```bash
  python -m parrotflume --key "your-api-key-here" --chat
  ```
- **`--model <model>`**: Set the model to use (e.g., `gpt-4o`, `deepseek-chat`). Example:
  ```bash
  python -m parrotflume --model gpt-4o --chat
  ```

#### Model Behavior Parameters
- **`--max-tokens <max>`**: Set the maximum number of tokens to generate. Example:
  ```bash
  python -m parrotflume --max-tokens 100 --chat
  ```
- **`--warmth <temperature>`**: Set the temperature for the model (controls randomness, 0.0 to 2.0). Example:
  ```bash
  python -m parrotflume --warmth 0.7 --chat
  ```

#### Output Formatting Parameters
- **`--markdown`**: Enable Markdown rendering in the output. Example:
  ```bash
  python -m parrotflume --markdown --chat
  ```
- **`--no-markdown`**: Disable Markdown rendering in the output. Example:
  ```bash
  python -m parrotflume --no-markdown --chat
  ```
- **`--color`**: Enable colored output. Example:
  ```bash
  python -m parrotflume --color --chat
  ```
- **`--no-color`**: Disable colored output. Example:
  ```bash
  python -m parrotflume --no-color --chat
  ```
- **`--latex`**: Enable LaTeX replacement in the output. Example:
  ```bash
  python -m parrotflume --latex --chat
  ```
- **`--no-latex`**: Disable LaTeX replacement in the output. Example:
  ```bash
  python -m parrotflume --no-latex --chat
  ```
- **`--func`**: Enable function calling (for supported models). Example:
  ```bash
  python -m parrotflume --func --chat
  ```
- **`--no-func`**: Disable function calling. Example:
  ```bash
  python -m parrotflume --no-func --chat
  ```

## Environment Variable

You can set the `OPENAI_API_KEY` environment variable to avoid hardcoding your API key in the configuration file:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

This will override any API key specified in the configuration file.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

ParrotFlume is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- The name "ParrotFlume" is inspired by the concept of "stochastic parrots" in LLMs and the idea of flumes as input/output pipes.
- Special thanks to the OpenAI community for their continuous support and development of powerful language models.

---

Enjoy using ParrotFlume! For any questions or issues, please refer to the [GitHub issues page](https://github.com/yourusername/ParrotFlume/issues).