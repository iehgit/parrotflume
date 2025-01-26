# ParrotFlume

**ParrotFlume** is a versatile command-line tool for using pipes (flumes) with LLMs (stochastic parrots[¹](https://dl.acm.org/doi/10.1145/3442188.3445922)), through OpenAI-compatible APIs. It also comes with a neat interactive chat CLI. Feed it through stdin, files, or keyboard input. Use it for scripting, single tasks or idle chitchat.

## Operation Modes

- **Transform Mode**: Transform data from stdin or file content based on a given instruction.
- **Perform Mode**: Prompt the LLM about data from stdin or file content.
- **Interactive Chat Mode**: Engage in a conversation with the LLM.
- **One-Shot Mode**: Provide a single prompt and receive an immediate, complete response from the LLM.

## Chat features
- **Model switching**: Seamlessly switch between API providers and models during a conversation. Ask a second model to verify the reply of the first. Then ask a third who of the previous two was right.
- **Markdown, LaTeX, and Color Support**: Enhanced output formatting with ANSI escape sequences and LaTeX to Unicode replacement.
- **Function Calling**: Let the LLM evaluate mathematical expressions, solve equations, and more using built-in functions.
- **Auto-Completion**: Enjoy tab-completion for API providers, models, and file paths when using commands in the interactive chat interface.

![screenshot](screenshot.png)

## Installation
### From Git
To install ParrotFlume, clone the repository and install the package alongside its required dependencies:

```bash
git clone https://github.com/iehgit/parrotflume.git
cd parrotflume
pip install .
```
### From PyPi
To install ParrotFlume directly from [PyPi](https://pypi.org/project/parrotflume/), alongside its dependencies: 
```bash
pip install parrotflume
```
### Building a .deb
It is also possible to build and install it as a debian package:
```bash
git clone https://github.com/iehgit/parrotflume.git
cd parrotflume
dpkg-buildpackage -us -uc
# dpkg -i ../parrotflume_*.*.*_all.deb
```

## Configuration

ParrotFlume uses a TOML configuration file to manage API providers, model settings, and global options.
Ask `parrotflume --help` about the place to put it for your operating system.  
For example, in Linux, the file should be placed there:
```
~/.config/parrotflume/parrotflume.config.toml
```
The configuration file is optional if all needed parameters (url, key, model) are provided otherwise.

### Example Configuration (`parrotflume.config.toml`)

```toml
# Example TOML configuration file for parrotflume

[global_options]
temperature = 0.1     # Default temperature for the model
max_tokens = 4096     # Maximum number of tokens to generate
markdown = true       # Enable markdown rendering
latex = true          # Enable LaTeX replacement
color = true          # Enable colored output
color_name = "green"  # ANSI name for colored output

# API providers, the first in the list is used as default
[[api_providers]]
name = "openai"
base_url = "https://api.openai.com/v1/"
api_key = "<yourapikeyhere>"
model = "gpt-4o"
func = true  # Enable function calling

[[api_providers]]
name = "deepseek"
base_url = "https://api.deepseek.com/v1/"
api_key = "<yourapikeyhere>"
model = "deepseek-chat"
func = true  # Enable function calling

[[api_providers]]
name = "llama.cpp"
base_url = "http://localhost:8080/v1/"
api_key = "sk-no-key-required"  # not used, NOT allowed to be empty for llama.cpp
model = ""   # not used, allowed to be empty for llama.cpp
func = false  # Disable function calling, not yet supported by llama.cpp

[[api_providers]]
name = "openrouter"
base_url = "https://openrouter.ai/api/v1/"
api_key = "<yourapikeyhere>"
model = "anthropic/claude-3.5-sonnet:beta"
func = true  # Enable function calling
```

## Usage

Command line parameters supersede environment variables. Environment variables supersede configuration file settings.

### Command-Line Parameters

#### General Parameters
- **`--chat`**: Start an interactive chat session with the LLM.
- **`--oneshot "<prompt>"`**: Provide a single prompt and get an immediate response. Example:
  ```bash
  parrotflume --oneshot "What is the meaning of life, the universe, and everything?"
  ```
- **`--transform "<prompt>" [filename]`**: Transform the content of a file using a prompt. If no filename is provided, reads from `stdin`. Examples:
  ```bash
  parrotflume --transform "Convert all strings to uppercase" input.txt 
  ```
  ```bash
  parrotflume --transform "Fix all syntax errors" < input.txt > output.txt
  ```
- **`--perform "<prompt>" [filename]`**: Perform a task on the content of a file. If no filename is provided, reads from `stdin`. Examples:
  ```bash
  parrotflume --perform "Extract all email addresses" input.txt > emails.txt
  ```
  ```bash
  dmesg | parrotflume --perform "Explain all ACPI errors" 
  ```
- **`--list`**: List all available models from the configured API provider. Example:
  ```bash
  parrotflume --list
  ```

#### API Configuration Parameters
- **`--api-provider <provider>`**: Set the API provider (e.g., `openai`, `deepseek`, `llama.cpp`). Example:
  ```bash
  parrotflume --api-provider openai --chat
  ```
- **`--base-url <url>`**: Set the base URL for the API provider. Example:
  ```bash
  parrotflume --base-url "https://api.openai.com/v1/" --chat
  ```
- **`--key <key>`**: Set the API key for the API provider. Example:
  ```bash
  parrotflume --key "your-api-key-here" --chat
  ```
- **`--model <model>`**: Set the model to use (e.g., `gpt-4o`, `deepseek-chat`). Example:
  ```bash
  parrotflume --model gpt-4o --chat
  ```

#### Model Behavior Parameters
- **`--max-tokens <max>`**: Set the maximum number of tokens to generate. Example:
  ```bash
  parrotflume --max-tokens 2048 --chat
  ```
- **`--warmth <temperature>`**: Set the temperature for the model. Example:
  ```bash
  parrotflume --warmth 0.7 --chat
  ```

#### Output Formatting Parameters
These apply to chat and one-shot modes.

- **`--markdown`**: Enable Markdown rendering in the output.
- **`--no-markdown`**: Disable Markdown rendering in the output.
- **`--color`**: Enable colored output.
- **`--no-color`**: Disable colored output.
- **`--latex`**: Enable LaTeX replacement in the output.
- **`--no-latex`**: Disable LaTeX replacement in the output. 

#### Function Calling Parameters

- **`--func`**: Enable function calling.
- **`--no-func`**: Disable function calling.

This provides the LLM with some mathematical tools, the current date, and a way to sift through text using regex.  
The function calling feature requires support from both the API provider and the LLM.

#### JSON Parameter

- **`--json`**: Enforces JSON output

### Environment Variable

You can set the `OPENAI_API_KEY` environment variable to avoid passing your API key in a command line where it might be logged:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

This will override any API key specified in the configuration file.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

ParrotFlume is licensed under an extended MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- The name "ParrotFlume" is inspired by the exhaustion of all slightly less silly names by different projects, and the title of a paper called "On the Dangers of Stochastic Parrots: Can Language Models Be Too Big? 🦜"[¹](https://dl.acm.org/doi/10.1145/3442188.3445922).
- Special thanks to the developers of [llama.cpp](https://github.com/ggerganov/llama.cpp), which works nicely as a backend for ParrotFlume.

## FAQ

### Is it dead?

No no he's not dead, he's, he's restin'! Remarkable bird, the Norwegian Blue, idn'it, ay? Beautiful plumage!

---

Enjoy using ParrotFlume! For any questions or issues, please refer to the [GitHub issues page](https://github.com/iehgit/parrotflume/issues).