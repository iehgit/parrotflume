from dataclasses import dataclass


@dataclass
class Config:
    api_provider: str = None
    base_url: str = None
    api_key: str = None
    model: str = None
    temperature: float = 0.0
    max_tokens: int = 4096
    do_markdown: bool = True
    do_latex: bool = True
    do_color: bool = False
    color: str = "bright_yellow"
    func: bool = True
    json: bool = False
    prefix: str = None