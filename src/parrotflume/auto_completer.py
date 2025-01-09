import os
import readline
from functools import lru_cache
import openai


class AutoCompleter:
    def __init__(self, api_providers):
        self.api_providers = api_providers

    def auto_complete(self, text, state):
        line = readline.get_line_buffer()

        # Handle API provider completion for "/p " commands
        if line.startswith("/p "):
            # Extract the prefix after "/p "
            prefix = line[3:].lstrip()

            # Filter providers that match the prefix
            matches = [provider for provider in self.api_providers if provider.startswith(prefix)]

            # Return the match corresponding to the state
            if state < len(matches):
                return matches[state]
            else:
                return None

        # Handle model completion for "/m " commands
        elif line.startswith("/m "):
            # Extract the prefix after "/m "
            prefix = line[3:].lstrip()

            # Get available models from the OpenAI API
            model_ids = self.get_models(openai.base_url)

            # Filter models that match the prefix
            matches = [model for model in model_ids if model.startswith(prefix)]

            # Return the match corresponding to the state
            if state < len(matches):
                return matches[state]
            else:
                return None

        # Check if the input starts with a file command
        if not any(line.startswith(f"/{cmd} ") for cmd in ("c", "d", "f", "u")):
            return None

        # Expand ~ to the user's home directory
        if '~' in text:
            text = os.path.expanduser(text)

        # Get the directory and prefix
        directory, prefix = os.path.split(text)

        # If no directory is specified, use the current directory
        if not directory:
            directory = '.'

        # Get all files and directories in the specified directory
        try:
            files = os.listdir(directory)
        except OSError:
            return None

        # Filter files that match the prefix
        matches = [f for f in files if f.startswith(prefix)]

        # Add the directory back to the matches
        matches = [os.path.join(directory, f) for f in matches]

        # Return the match corresponding to the state
        if state < len(matches):
            return matches[state]
        else:
            return None

    def setup(self):
        readline.set_completer(self.auto_complete)
        readline.parse_and_bind("tab: complete")
        readline.set_completer_delims(' \t\n')  # Treat spaces and tabs as delimiters

    @lru_cache
    def get_models(self, _cache_key):
        # noinspection PyBroadException
        try:
            return [model.id for model in openai.models.list()]
        except Exception:
            return []
