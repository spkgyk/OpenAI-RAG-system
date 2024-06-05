from typing import Any, Dict
import argparse
import yaml
import os


class ConfigLoader:
    def __init__(self, config_file: str) -> None:
        self.config_file = config_file
        self.config = self._load_config()
        self.args = self._parse_arguments()
        self._apply_overrides()
        self.config["openai_key"] = os.getenv("OPENAI_KEY")

    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_file, "r") as file:
            return yaml.safe_load(file)

    def _parse_arguments(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Load model from argument or config file")
        parser.add_argument("--model", type=str, help="The name of the model")
        parser.add_argument("--top-k", type=int, help="Top-K results are used for the augmented generation")
        return parser.parse_args()

    def _apply_overrides(self) -> None:
        self.config["model"] = self.args.model if self.args.model else self.config.get("model", "llama3")
        self.config["top_k"] = self.args.top_k if self.args.top_k else self.config.get("top_k", 5)

    def get_config(self) -> Dict[str, Any]:
        return self.config


if __name__ == "__main__":
    config_loader = ConfigLoader("default.yaml")
    config = config_loader.get_config()
    print(f"Using model: {config['model']} with top-{config['top_k']}.")
