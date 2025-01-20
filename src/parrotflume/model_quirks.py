no_function_call = ["o1-preview", "deepseek-reasoner"]  # supports no function calling at all
no_function_call_looping = ["deepseek-"]  # runs into endless loops if looping is allowed
no_system = ["o1-preview"]  # does not know role "system"
developer = ["o1", "o3"]  # "system" has been renamed to "developer"
max_completion_tokens = ["o1", "o3"]  # uses "max_completion_tokens" instead of "max_tokens"
no_temperature = ["o1", "o3", "deepseek-reasoner"]  # has no temperature setting
no_json = ["deepseek-reasoner"]  # does not support json response format
