"""Generate finetune dynamics plots from training CSV data."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

out_dir = r"D:\00workspace\academic_relevant\25-26-spring\AIAA3201 (L01) - Introduction to Computer Vision\Project\Project\NLP_group_project\figures"
os.makedirs(out_dir, exist_ok=True)

# SQuAD data
squad_steps = list(range(20, 36401, 20))
squad_loss = [4.745,4.4125,4.2336,3.287,2.9618,2.5904,2.2055,2.0669,1.7868,1.5349,1.545,1.5448,1.5394,1.088,1.2928,1.3779,1.1272,1.3631,1.291,1.494,1.1893,1.0956,0.9194,1.1079,1.0809,1.0747,1.2509,0.8828,1.0654,1.2249,1.0166,1.0445,1.01,0.9499,1.03,1.0551,0.9983,1.1511,1.1194,0.9951,0.8961,1.0876,0.9832,0.9472,1.0225,0.9072,1.0259,0.9543,0.979,0.7843,0.8814,0.8336,1.0101,1.0313,0.904,0.8808,0.8962,1.0224,0.9221,0.9852,0.8298,0.8386,0.9274,0.6793,1.133,0.9007,0.8265,0.9079,1.1029,1.0149,0.9459,1.0814,0.8591,1.0035,0.8293,0.7497,0.9927,0.7942,0.9645,0.9399,0.9172,0.8184,0.9388,0.9942,0.7594,1.0566]
# Tail has 1000+ more points, approximate
squad_steps_full = [i * 20 for i in range(1, 1821)]
# Only first ~1750 pts shown for readability

# SciQ data
sciq_steps = [20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400,420,440,460,480,500,520,540,560,580,600,620,640,660,680,700,720,740,760,780,800,820,840,860,880,900,920,940,960,980,1000,1020,1040,1060,1080,1100,1120,1140,1160,1180,1200,1220,1240,1260,1280,1300,1320,1340,1360,1380,1400,1420,1440,1460,1480,1500,1520,1540,1560,1580,1600,1620,1640,1660,1680,1700,1720]
sciq_loss = [5.9998,3.1474,0.9465,0.773,0.7372,0.7197,0.7618,0.7678,0.7604,0.6931,0.7553,0.7279,0.7564,0.718,0.721,0.7241,0.7362,0.716,0.7192,0.6578,0.6175,0.6354,0.6754,0.5546,0.6111,0.5291,0.465,0.5104,0.4851,0.3939,0.353,0.3645,0.4358,0.5553,0.2905,0.4615,0.4372,0.2422,0.3903,0.3928,0.3231,0.3719,0.3173,0.3288,0.4107,0.2549,0.3107,0.2502,0.3917,0.3647,0.2391,0.3342,0.2343,0.3488,0.2942,0.3082,0.251,0.4115,0.2761,0.24,0.3295,0.3594,0.3051,0.3609,0.3772,0.2172,0.2698,0.3214,0.3388,0.3532,0.2701,0.2239,0.1958,0.1747,0.326,0.3592,0.2606,0.2944,0.2973,0.3656,0.3341,0.2482,0.3646,0.1595,0.3675,0.1887]

# SQuAD faithfulness
squad_f_steps = [200,400,600,800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,3200,3400,3600,3800,4000,4200,4400,4600,4800,5000,5200,5400,5600,5800,6000,6200,6400,6600,6800]
squad_morf = [0.0087,0.0785,0.1345,0.1386,0.2910,0.1599,0.1336,0.2769,0.2622,0.2878,0.0808,0.0823,0.2553,0.4162,0.3160,0.1510]  # first 16 only for plot clarity
squad_rand_morf = [0.0075,0.0314,0.0559,0.0765,0.1169,0.0679,0.0838,0.1596,0.2018,0.2188,0.0671,0.0684,0.1684,0.2729,0.2219,0.0902]
squad_lerf = [0.0044,0.0058,0.0104,0.0100,0.0130,0.0140,0.0107,0.0138,0.0152,0.0127,0.0111,0.0104,0.0157,0.0193,0.0127,0.0145]
squad_rand_lerf = [0.0075,0.0419,0.0508,0.0546,0.1258,0.0883,0.0784,0.1381,0.1722,0.1082,0.0873,0.0646,0.1026,0.2950,0.1411,0.1024]

# SciQ faithfulness
sciq_f_steps = [200,400,600,800,1000,1200,1400,1600,1800,2000,2200]
sciq_morf = [0.1856,0.3369,0.2242,0.2880,0.4524,0.4014,0.3061,0.3588,0.3077,0.3937,0.3353]
sciq_rand_morf = [0.0772,0.0964,0.1320,0.2387,0.2648,0.3186,0.1475,0.2846,0.2638,0.2531,0.2578]
sciq_lerf = [0.0181,0.0176,0.0454,0.0265,0.0742,0.0512,0.0276,0.0273,0.0366,0.0378,0.0334]
sciq_rand_lerf = [0.0914,0.1279,0.2589,0.1861,0.3041,0.3018,0.2924,0.2218,0.3089,0.2322,0.2327]

# SQuAD accuracy
squad_acc_steps = [200,400,600,800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,3200,3400,3600,3800,4000,4200,4400,4600,4800,5000,5200,5400,5600,5800,6000,6200,6400,6600,6800]
squad_token_acc = [0.7506,0.7964,0.8106,0.8180,0.8309,0.8294,0.8307,0.8446,0.8355,0.8423,0.8315,0.8378,0.8380,0.8481,0.8484,0.8289,0.8425,0.8493,0.8446,0.8479,0.8474,0.8540,0.8460,0.8388,0.8383,0.8465,0.8448,0.8449,0.8538,0.8389,0.8581,0.8521,0.8562,0.8545]

# SciQ accuracy
sciq_acc_steps = [200,400,600,800,1000,1200,1400,1600,1800,2000,2200]
sciq_token_acc = [0.637,0.757,0.89,0.907,0.9175,0.9015,0.934,0.9315,0.933,0.935,0.9345]

# ── Figure 1: Loss Curves ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
# SQuAD
steps_s = squad_steps[:len(squad_loss)]
ax1.plot(steps_s, squad_loss, 'b-', linewidth=0.8, alpha=0.7)
ax1.set_xlabel('Training Step')
ax1.set_ylabel('Loss')
ax1.set_title('SQuAD Fine-tuning Loss')
ax1.grid(True, alpha=0.3)
# SciQ
ax2.plot(sciq_steps, sciq_loss, 'r-', linewidth=0.8, alpha=0.7)
ax2.set_xlabel('Training Step')
ax2.set_ylabel('Loss')
ax2.set_title('SciQ Fine-tuning Loss')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'fig_loss_curves.png'), dpi=150)
plt.close()
print("Saved fig_loss_curves.png")

# ── Figure 2: Faithfulness AUC over Steps (single random baseline) ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
squad_rand = [(a+b)/2 for a,b in zip(squad_rand_morf, squad_rand_lerf)]
sciq_rand = [(a+b)/2 for a,b in zip(sciq_rand_morf, sciq_rand_lerf)]
# SQuAD
ax1.plot(squad_f_steps[:len(squad_morf)], squad_morf, 'b-o', markersize=3, label='MoRF (AttnLRP)')
ax1.plot(squad_f_steps[:len(squad_lerf)], squad_lerf, 'r-o', markersize=3, label='LeRF (AttnLRP)')
ax1.plot(squad_f_steps[:len(squad_rand)], squad_rand, 'k--', linewidth=1.5, label='Random')
ax1.set_xlabel('Training Step')
ax1.set_ylabel('AUC')
ax1.set_title('SQuAD: Faithfulness AUC')
ax1.legend(fontsize=7)
ax1.grid(True, alpha=0.3)
# SciQ
ax2.plot(sciq_f_steps, sciq_morf, 'b-o', markersize=3, label='MoRF (AttnLRP)')
ax2.plot(sciq_f_steps, sciq_lerf, 'r-o', markersize=3, label='LeRF (AttnLRP)')
ax2.plot(sciq_f_steps, sciq_rand, 'k--', linewidth=1.5, label='Random')
ax2.set_xlabel('Training Step')
ax2.set_ylabel('AUC')
ax2.set_title('SciQ: Faithfulness AUC')
ax2.legend(fontsize=7)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'fig_faithfulness_auc.png'), dpi=150)
plt.close()
print("Saved fig_faithfulness_auc.png")

# ── Figure 3: Token Accuracy over Steps ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(squad_acc_steps, squad_token_acc, 'g-', linewidth=1)
ax1.set_xlabel('Training Step')
ax1.set_ylabel('Token Accuracy')
ax1.set_title('SQuAD: Eval Token Accuracy')
ax1.grid(True, alpha=0.3)
ax2.plot(sciq_acc_steps, sciq_token_acc, 'g-', linewidth=1)
ax2.set_xlabel('Training Step')
ax2.set_ylabel('Token Accuracy')
ax2.set_title('SciQ: Eval Token Accuracy')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'fig_accuracy.png'), dpi=150)
plt.close()
print("Saved fig_accuracy.png")

# ── Figure 4: Comprehensiveness/Sufficiency Gaps over Steps ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
# SQuAD gaps
dc_s = [m - r for m, r in zip(squad_morf[:16], squad_rand)]
ds_s = [l - r for l, r in zip(squad_lerf[:16], squad_rand)]
ax1.plot(squad_f_steps[:16], dc_s, 'b-o', markersize=3, label='Comp Gap')
ax1.plot(squad_f_steps[:16], ds_s, 'r-o', markersize=3, label='Suff Gap')
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax1.set_xlabel('Training Step')
ax1.set_ylabel('Gap')
ax1.set_title('SQuAD: Comp & Suff Gaps')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

dc_q = [m - r for m, r in zip(sciq_morf, sciq_rand)]
ds_q = [l - r for l, r in zip(sciq_lerf, sciq_rand)]
ax2.plot(sciq_f_steps, dc_q, 'b-o', markersize=3, label='Comp Gap')
ax2.plot(sciq_f_steps, ds_q, 'r-o', markersize=3, label='Suff Gap')
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('Training Step')
ax2.set_ylabel('Gap')
ax2.set_title('SciQ: Comp & Suff Gaps')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'fig_gaps_over_steps.png'), dpi=150)
plt.close()
print("Saved fig_gaps_over_steps.png")
