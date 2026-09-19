from .tokenizer import GraphVQTokenizer


def build_model(cfg: dict) -> GraphVQTokenizer:
    """Build the model from a config dict (the ``model`` section of config.yaml)."""
    return GraphVQTokenizer(
        in_dim=cfg["in_dim"],
        hidden_dim=cfg["hidden_dim"],
        latent_dim=cfg["latent_dim"],
        num_layers=cfg["num_layers"],
        codebook_size=cfg["codebook_size"],
    )
