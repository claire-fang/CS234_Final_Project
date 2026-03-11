"""Generate extra result plots from train_metrics.json.

Usage:
    python plot_results.py
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

METRICS = 'checkpoints_blind/train_metrics.json'
OUT_DIR = 'checkpoints_blind'


def load():
    with open(METRICS) as f:
        return json.load(f)


def plot_f1_by_gold(m):
    """Grouped bar chart: retrieval F1 broken down by gold=2 vs gold=4."""
    bc = m['bc_retrieval']
    sft_dpo = m.get('sft_dpo_retrieval', {})
    ppo = m['ppo_retrieval']
    bl = {b['strategy']: b for b in m['baselines_retrieval']}

    strategies = (['Random (1)', 'Random (2)', 'Random (3)', 'Random (4)',
                   'Random (5)', 'Random (6)', 'Random (7)',
                   'Greedy (1)', 'Greedy (2)', 'Greedy (3)', 'Greedy (4)',
                   'Greedy (5)', 'Greedy (6)', 'Greedy (7)',
                   'BC-only', 'SFT+DPO', 'PPO (ours)'])
    gold2_f1, gold4_f1, overall_f1 = [], [], []
    for s in strategies:
        if s == 'BC-only':
            src = bc
        elif s == 'SFT+DPO':
            src = sft_dpo
        elif s == 'PPO (ours)':
            src = ppo
        else:
            src = bl[s]
        gold2_f1.append(src['gold_2']['f1'] * 100)
        gold4_f1.append(src['gold_4']['f1'] * 100)
        overall_f1.append(src['f1'] * 100)

    x = np.arange(len(strategies))
    w = 0.25
    fig, ax = plt.subplots(figsize=(18, 6))
    b1 = ax.bar(x - w, gold2_f1, w, label='Gold=2', color='#5B9BD5',
                edgecolor='white', linewidth=0.5)
    b2 = ax.bar(x, gold4_f1, w, label='Gold=4', color='#ED7D31',
                edgecolor='white', linewidth=0.5)
    b3 = ax.bar(x + w, overall_f1, w, label='Overall', color='#70AD47',
                edgecolor='white', linewidth=0.5)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8)

    ppo_gap = abs(gold2_f1[-1] - gold4_f1[-1])
    bc_gap = abs(gold2_f1[-2] - gold4_f1[-2])
    g5_idx = strategies.index('Greedy (5)')
    greedy5_gap = abs(gold2_f1[g5_idx] - gold4_f1[g5_idx])
    ax.annotate(
        f'PPO gap: {ppo_gap:.1f}%  |  BC gap: {bc_gap:.1f}%\n'
        f'(Greedy-5 gap: {greedy5_gap:.1f}%)',
        xy=(x[-1], max(gold2_f1[-1], gold4_f1[-1]) + 7),
        fontsize=9, ha='center', color='#C00000',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF2CC',
                  edgecolor='#C00000', alpha=0.8))

    ax.set_ylabel('Retrieval F1 (%)', fontsize=12)
    ax.set_title('Retrieval F1 by Gold Paragraph Count (Gold=2 vs Gold=4)',
                 fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=8, rotation=30, ha='right')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    path = f'{OUT_DIR}/f1_by_gold_count.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved {path}')


def plot_adaptive_reads(m):
    """Bar chart showing PPO/BC adapt read count to question difficulty."""
    bc = m['bc_retrieval']
    sft_dpo = m.get('sft_dpo_retrieval', {})
    ppo = m['ppo_retrieval']
    bl = {b['strategy']: b for b in m['baselines_retrieval']}

    strategies = (['Random (1)', 'Random (2)', 'Random (3)', 'Random (4)',
                   'Random (5)', 'Random (6)', 'Random (7)',
                   'Greedy (1)', 'Greedy (2)', 'Greedy (3)', 'Greedy (4)',
                   'Greedy (5)', 'Greedy (6)', 'Greedy (7)',
                   'BC-only', 'SFT+DPO', 'PPO (ours)'])
    gold2_reads, gold4_reads = [], []
    for s in strategies:
        if s == 'BC-only':
            src = bc
        elif s == 'SFT+DPO':
            src = sft_dpo
        elif s == 'PPO (ours)':
            src = ppo
        else:
            src = bl[s]
        gold2_reads.append(src['gold_2']['avg_reads'])
        gold4_reads.append(src['gold_4']['avg_reads'])

    x = np.arange(len(strategies))
    w = 0.32
    fig, ax = plt.subplots(figsize=(18, 5.5))
    b1 = ax.bar(x - w / 2, gold2_reads, w, label='Gold=2 questions',
                color='#5B9BD5', edgecolor='white', linewidth=0.5)
    b2 = ax.bar(x + w / 2, gold4_reads, w, label='Gold=4 questions',
                color='#ED7D31', edgecolor='white', linewidth=0.5)

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9,
                        fontweight='bold')

    ax.axhspan(1.5, 4.5, xmin=0.85, xmax=1.0, alpha=0.08, color='green')
    ax.annotate('Fixed reads\n(cannot adapt)',
                xy=(6, 5.1), fontsize=9, ha='center', color='#666666',
                style='italic')
    ax.annotate('Adaptive reads\n(learned policy)',
                xy=(14.5, 4.4), fontsize=9, ha='center', color='#2E7D32',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9',
                          edgecolor='#2E7D32', alpha=0.8))

    # Delta arrows for BC-only
    bc_idx = strategies.index('BC-only')
    ax.annotate('', xy=(bc_idx + 0.22, gold4_reads[bc_idx]),
                xytext=(bc_idx - 0.22, gold2_reads[bc_idx]),
                arrowprops=dict(arrowstyle='<->', color='#C00000', lw=1.8))
    bc_gap = gold4_reads[bc_idx] - gold2_reads[bc_idx]
    ax.text(bc_idx,
            (gold2_reads[bc_idx] + gold4_reads[bc_idx]) / 2 + 0.15,
            f'\u0394 = {bc_gap:.2f}', ha='center', fontsize=9,
            color='#C00000', fontweight='bold')

    # Delta arrows for PPO
    ppo_idx = strategies.index('PPO (ours)')
    ax.annotate('', xy=(ppo_idx + 0.22, gold4_reads[ppo_idx]),
                xytext=(ppo_idx - 0.22, gold2_reads[ppo_idx]),
                arrowprops=dict(arrowstyle='<->', color='#C00000', lw=1.8))
    ppo_gap = gold4_reads[ppo_idx] - gold2_reads[ppo_idx]
    ax.text(ppo_idx,
            (gold2_reads[ppo_idx] + gold4_reads[ppo_idx]) / 2 + 0.15,
            f'\u0394 = {ppo_gap:.2f}', ha='center', fontsize=9,
            color='#C00000', fontweight='bold')

    ax.set_ylabel('Average Number of Reads', fontsize=12)
    ax.set_title(
        'Adaptive Read Count: Learned Policies Read More for Harder Questions',
        fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=8, rotation=30, ha='right')
    ax.legend(fontsize=11, loc='upper left')
    ax.set_ylim(0, 8.0)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(y=2, color='#5B9BD5', linestyle=':', alpha=0.4, linewidth=1)
    ax.axhline(y=4, color='#ED7D31', linestyle=':', alpha=0.4, linewidth=1)
    ax.text(-0.4, 2.05, 'ideal for gold=2', fontsize=7,
            color='#5B9BD5', alpha=0.7)
    ax.text(-0.4, 4.05, 'ideal for gold=4', fontsize=7,
            color='#ED7D31', alpha=0.7)
    plt.tight_layout()
    path = f'{OUT_DIR}/adaptive_reads.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved {path}')


def plot_f1_comparison_bar(m):
    """Overall F1 bar chart with gold-count breakdown (mirrors train_only.py Plot 4)."""
    bc = m['bc_retrieval']
    sft_dpo = m.get('sft_dpo_retrieval', {})
    ppo = m['ppo_retrieval']
    bl = {b['strategy']: b for b in m['baselines_retrieval']}

    strategies = ([f'Random ({k})' for k in range(1, 8)]
                  + [f'Greedy ({k})' for k in range(1, 8)]
                  + ['BC-only', 'SFT+DPO', 'PPO (ours)'])

    # Collect per-strategy data dicts (same structure as all_ret in train_only)
    bar_data = []
    for s in strategies:
        if s == 'BC-only':
            src = bc
        elif s == 'SFT+DPO':
            src = sft_dpo
        elif s == 'PPO (ours)':
            src = ppo
        else:
            src = bl[s]
        bar_data.append(src)

    gold_counts = sorted({int(k.split('_')[1]) for d in bar_data
                          for k in d if k.startswith('gold_')})

    bar_f1_all = [d['f1'] * 100 for d in bar_data]
    x = np.arange(len(strategies))
    width = 0.25

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.bar(x, bar_f1_all, width, label='All', color='#2196F3', alpha=0.8)
    colors = ['#FF9800', '#4CAF50', '#9C27B0', '#F44336']
    for gc_i, gc in enumerate(gold_counts):
        key = f'gold_{gc}'
        bar_f1_gc = [d.get(key, {}).get('f1', 0) * 100 for d in bar_data]
        offset = width * (gc_i + 1)
        ax.bar(x + offset, bar_f1_gc, width,
               label=f'Gold={gc}', color=colors[gc_i % len(colors)], alpha=0.8)

    ax.set_xlabel('Strategy')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Comparison by Gold Count')
    ax.set_xticks(x + width * len(gold_counts) / 2)
    ax.set_xticklabels(strategies, rotation=30, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    path = f'{OUT_DIR}/f1_comparison_bar.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {path}')


if __name__ == '__main__':
    print('Generating extra plots...')
    m = load()
    plot_f1_by_gold(m)
    plot_adaptive_reads(m)
    plot_f1_comparison_bar(m)
    print('Done.')
