"""Validation-selected training for the modern student model."""
import argparse
import json
import math
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from common import PROTOCOL, ROOT, autocast, device_metrics, load_data, make_model, setup, sha
from evaluate import score


def main():
    total_started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation', default='student_v4')
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--run-dir', required=True, type=Path)
    p.add_argument('--device', default='cuda')
    p.add_argument('--precision', choices=['auto', 'fp32', 'bf16'], default='bf16')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=12000)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--lr', type=float, default=4e-4)
    p.add_argument('--min-lr-ratio', type=float, default=.1)
    p.add_argument('--warmup', type=int, default=200)
    p.add_argument('--weight-decay', type=float, default=.1)
    args = p.parse_args()
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        p.error('Use a new empty run directory.')
    if min(args.steps, args.batch_size, args.eval_every) < 1:
        p.error('steps, batch-size and eval-every must be positive')
    device, precision = setup(args.device, args.precision, args.threads)
    torch.manual_seed(args.seed)
    prepared = time.perf_counter()
    data = load_data()
    config = json.loads(args.config.read_text())
    model, implementation_sha = make_model(args.implementation, config, device)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    optimizer = torch.optim.AdamW([
        {'params': decay, 'weight_decay': args.weight_decay},
        {'params': no_decay, 'weight_decay': 0.},
    ], lr=args.lr, betas=(.9, .95), eps=1e-8)
    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)
    offsets = torch.arange(257, device=device)
    best_bpb = math.inf
    best_step = 0
    best_state = None
    history, validation_history = [], []
    intermediate_validation_seconds = 0.
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter() - prepared
    started = time.perf_counter()
    for step in range(args.steps):
        starts = torch.randint(len(tokens) - 257, (args.batch_size,), generator=rng).to(device)
        batch = tokens[starts[:, None] + offsets]
        progress = step / max(1, args.steps - 1)
        cosine = .5 * (1 + math.cos(math.pi * progress))
        learning_rate = args.lr * min(1., (step + 1) / args.warmup) * (
            args.min_lr_ratio + (1 - args.min_lr_ratio) * cosine)
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(logits.flatten(0, 1).float(), batch[:, 1:].flatten())
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if (step + 1) % 100 == 0 or step + 1 == args.steps:
            row = {'step': step + 1, 'loss': loss.item(), 'lr': learning_rate,
                   'grad_norm': float(grad_norm),
                   'seconds': time.perf_counter() - started - intermediate_validation_seconds}
            history.append(row)
            print(json.dumps(row), flush=True)
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            validation = score(model, *data['validation'], device, 'fp32')
            validation.pop('window_nll_nats')
            intermediate_validation_seconds += validation['seconds']
            row = {'step': step + 1, **validation}
            validation_history.append(row)
            if validation['bpb'] < best_bpb:
                best_bpb, best_step = validation['bpb'], step + 1
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                torch.save({'step': best_step, 'bpb': best_bpb, 'model': best_state},
                           args.run_dir/'best_intermediate.pt')
            print(json.dumps({'validation': row, 'best_step': best_step, 'best_bpb': best_bpb}), flush=True)
    if best_state is None:
        raise RuntimeError('No validation checkpoint was selected.')
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - started - intermediate_validation_seconds
    model.load_state_dict(best_state)
    selected_validation = score(model, *data['validation'], device, 'fp32')
    selected_validation.pop('window_nll_nats')
    checkpoint = args.run_dir/'checkpoint.pt'
    torch.save({'protocol': PROTOCOL, 'implementation': args.implementation,
                'config': config, 'model': best_state, 'seed': args.seed,
                'train_tokens': args.steps * args.batch_size * 256,
                'selected_step': best_step,
                'selected_train_tokens': best_step * args.batch_size * 256,
                'optimizer_recipe': {'lr': args.lr, 'min_lr_ratio': args.min_lr_ratio,
                                     'warmup': args.warmup, 'weight_decay': args.weight_decay,
                                     'betas': [.9, .95]}}, checkpoint)
    (args.run_dir/'best_intermediate.pt').unlink()
    result = {'protocol': PROTOCOL, 'implementation': args.implementation,
              'config': config, 'seed': args.seed,
              'parameters': sum(p.numel() for p in model.parameters()),
              'precision': precision, 'planned_train_tokens': args.steps * args.batch_size * 256,
              'selected_train_tokens': best_step * args.batch_size * 256,
              'selected_step': best_step, 'best_validation': selected_validation,
              'preparation_seconds': preparation_seconds, 'train_seconds': train_seconds,
              'history': history, 'validation_history': validation_history,
              'intermediate_validation_seconds': intermediate_validation_seconds,
              'process_seconds': time.perf_counter() - total_started,
              'torch_version': str(torch.__version__), 'threads': args.threads,
              'checkpoint_sha256': sha(checkpoint),
              'implementation_sha256': implementation_sha, **device_metrics(device)}
    (args.run_dir/'metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result | {'history': [], 'validation_history': []}, indent=2), flush=True)


if __name__ == '__main__':
    main()
