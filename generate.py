"""
Generate text from a trained NanoGPT checkpoint.
"""
import argparse
import torch
from src.model import NanoGPT
from src.utils import get_device


def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if ckpt.get("tokenizer") != "utf8_bytes_v1":
        raise ValueError("Checkpoint is not marked as UTF-8. Retrain with the corrected train.py.")
    cfg = ckpt.get("config", {})
    model = NanoGPT(
        vocab_size=cfg.get("vocab_size", 256),
        block_size=cfg.get("block_size", 128),
        dim=cfg.get("dim", 128),
        n_layers=cfg.get("n_layers", 5),
        n_heads=cfg.get("n_heads", 4),
        dropout=0.0,  # no dropout at inference
    )
    # handle both plain state_dict and our wrapped checkpoint
    state = ckpt["model"] if "model" in ckpt else ckpt
    # strip possible torch.compile prefixes
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def bytes_to_text(ids) -> str:
    return bytes(int(i) for i in ids).decode("utf-8", errors="replace")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints_utf8/nanogpt_best.pt")
    p.add_argument("--prompt", default="Жил-был", help="starting text (UTF-8)")
    p.add_argument("--max_tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = get_device()
    torch.manual_seed(args.seed)

    model = load_model(args.ckpt, device)
    print(f"Loaded {args.ckpt} on {device}")

    # encode prompt the same way as training
    prompt_ids = list(args.prompt.encode("utf-8"))
    if not prompt_ids:
        p.error("--prompt must not be empty")
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    out = model.generate(
        idx,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    text = bytes_to_text(out[0].tolist())
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


if __name__ == "__main__":
    main()
