#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


def find_latest_trainer_state(output_dir: Path) -> Path:
    candidates = sorted(output_dir.glob('checkpoint-*/trainer_state.json'))
    if not candidates:
        raise FileNotFoundError(f'No trainer_state.json found under {output_dir}/checkpoint-*')

    def step_of(path: Path) -> int:
        m = re.search(r'checkpoint-(\d+)', str(path))
        return int(m.group(1)) if m else -1

    return max(candidates, key=step_of)


def main():
    parser = argparse.ArgumentParser(description='Plot train/eval loss from Hugging Face trainer_state.json')
    parser.add_argument('--output_dir', required=True, help='Training output directory (contains checkpoint-*/trainer_state.json)')
    parser.add_argument('--trainer_state', default='', help='Optional explicit trainer_state.json path')
    parser.add_argument('--out_png', default='', help='Output PNG path (default: <output_dir>/train_eval_loss_plot.png)')
    parser.add_argument('--out_csv', default='', help='Output CSV path (default: <output_dir>/train_eval_loss_points.csv)')
    parser.add_argument('--faith_out_png', default='', help='Faithfulness PNG path (default: <output_dir>/faithfulness_auc_plot.png)')
    parser.add_argument('--faith_out_csv', default='', help='Faithfulness CSV path (default: <output_dir>/faithfulness_auc_points.csv)')
    parser.add_argument('--acc_out_png', default='', help='Accuracy PNG path (default: <output_dir>/eval_accuracy_plot.png)')
    parser.add_argument('--acc_out_csv', default='', help='Accuracy CSV path (default: <output_dir>/eval_accuracy_points.csv)')
    parser.add_argument('--title', default='', help='Plot title')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    trainer_state_path = Path(args.trainer_state) if args.trainer_state else find_latest_trainer_state(output_dir)

    out_png = Path(args.out_png) if args.out_png else output_dir / 'train_eval_loss_plot.png'
    out_csv = Path(args.out_csv) if args.out_csv else output_dir / 'train_eval_loss_points.csv'
    faith_out_png = Path(args.faith_out_png) if args.faith_out_png else output_dir / 'faithfulness_auc_plot.png'
    faith_out_csv = Path(args.faith_out_csv) if args.faith_out_csv else output_dir / 'faithfulness_auc_points.csv'
    acc_out_png = Path(args.acc_out_png) if args.acc_out_png else output_dir / 'eval_accuracy_plot.png'
    acc_out_csv = Path(args.acc_out_csv) if args.acc_out_csv else output_dir / 'eval_accuracy_points.csv'

    obj = json.loads(trainer_state_path.read_text(encoding='utf-8'))
    log = obj.get('log_history', [])

    train = [(x.get('step'), x.get('loss')) for x in log if 'loss' in x and 'step' in x]
    evals = [(x.get('step'), x.get('eval_loss')) for x in log if 'eval_loss' in x and 'step' in x]
    faith = [
        (
            x.get('step'),
            x.get('eval_auc_morf'),
            x.get('eval_auc_lerf'),
            x.get('eval_auc_random_morf'),
            x.get('eval_auc_random_lerf'),
            x.get('eval_faithfulness_samples'),
        )
        for x in log
        if 'eval_auc_morf' in x and 'step' in x
    ]
    acc = [
        (
            x.get('step'),
            x.get('eval_token_acc'),
            x.get('eval_seq_acc'),
        )
        for x in log
        if 'eval_token_acc' in x and 'step' in x
    ]

    if not train and not evals:
        raise ValueError(f'No train/eval loss found in {trainer_state_path}')

    title = args.title or f'Loss Curve ({output_dir.name})'

    plt.figure(figsize=(9, 5))
    if train:
        plt.plot([s for s, _ in train], [v for _, v in train], label='train_loss', marker='o', markersize=2.5, linewidth=1.3)
    if evals:
        plt.plot([s for s, _ in evals], [v for _, v in evals], label='eval_loss', marker='s', markersize=4, linewidth=1.4)
    else:
        plt.text(0.02, 0.95, 'No eval_loss logged', transform=plt.gca().transAxes, fontsize=9, va='top')

    plt.title(title)
    plt.xlabel('Global Step')
    plt.ylabel('Loss')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)

    with out_csv.open('w', encoding='utf-8') as f:
        f.write('type,step,loss\n')
        for s, v in train:
            f.write(f'train,{s},{v}\n')
        for s, v in evals:
            f.write(f'eval,{s},{v}\n')

    # Faithfulness AUC plot (new PNG)
    if faith:
        plt.figure(figsize=(10, 5.5))
        steps = [x[0] for x in faith]
        morf = [x[1] for x in faith]
        lerf = [x[2] for x in faith]
        random_morf = [x[3] for x in faith]
        random_lerf = [x[4] for x in faith]

        plt.plot(steps, morf, marker='o', linewidth=1.5, label='eval_auc_morf (lower better)')
        plt.plot(steps, random_morf, marker='o', linestyle='--', linewidth=1.2, label='eval_auc_random_morf')
        plt.plot(steps, lerf, marker='s', linewidth=1.5, label='eval_auc_lerf (higher better)')
        plt.plot(steps, random_lerf, marker='s', linestyle='--', linewidth=1.2, label='eval_auc_random_lerf')
        plt.title(f'Faithfulness AUC Curve ({output_dir.name})')
        plt.xlabel('Global Step')
        plt.ylabel('AUC')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(faith_out_png, dpi=180)

        with faith_out_csv.open('w', encoding='utf-8') as f:
            f.write('step,eval_auc_morf,eval_auc_lerf,eval_auc_random_morf,eval_auc_random_lerf,eval_faithfulness_samples\n')
            for row in faith:
                f.write(','.join(str(v) for v in row) + '\n')

    # Accuracy plot (token/sequence)
    if acc:
        plt.figure(figsize=(10, 5.2))
        steps = [x[0] for x in acc]
        token_acc = [x[1] for x in acc]
        seq_acc = [x[2] for x in acc]

        plt.plot(steps, token_acc, marker='o', linewidth=1.5, label='eval_token_acc')
        plt.plot(steps, seq_acc, marker='s', linewidth=1.5, label='eval_seq_acc')
        plt.title(f'Eval Accuracy Curve ({output_dir.name})')
        plt.xlabel('Global Step')
        plt.ylabel('Accuracy')
        plt.ylim(0.0, 1.0)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(acc_out_png, dpi=180)

        with acc_out_csv.open('w', encoding='utf-8') as f:
            f.write('step,eval_token_acc,eval_seq_acc\n')
            for row in acc:
                f.write(','.join(str(v) for v in row) + '\n')

    print(json.dumps({
        'trainer_state': str(trainer_state_path),
        'train_points': len(train),
        'eval_points': len(evals),
        'faith_points': len(faith),
        'acc_points': len(acc),
        'out_png': str(out_png),
        'out_csv': str(out_csv),
        'faith_out_png': str(faith_out_png) if faith else '',
        'faith_out_csv': str(faith_out_csv) if faith else '',
        'acc_out_png': str(acc_out_png) if acc else '',
        'acc_out_csv': str(acc_out_csv) if acc else ''
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
