# FINAL results (Kaggle figures-and-audits Version 2, 22 Nov 2026 ~18:07 ET)
EfficientNet-B3 fp32 predictions for all 4 runs; non-finite arrays: 0.

## Discrimination (mean ± SD, 3 seeds)
| Model | HAM bal acc | HAM AUROC | BCN | MILK | PAD |
| ConvNeXt-T | 0.779 ± 0.026 | 0.911 ± 0.009 | 0.766 ± 0.024 | 0.774 ± 0.015 | 0.740 ± 0.023 |
| ResNet-50 | 0.696 ± 0.026 | 0.909 ± 0.017 | 0.696 ± 0.036 | 0.701 ± 0.013 | 0.728 ± 0.037 |
| Xception | 0.663 ± 0.015 | 0.894 ± 0.006 | 0.658 ± 0.031 | 0.681 ± 0.024 | 0.699 ± 0.068 |
| EfficientNet-B3 | 0.576 ± 0.022 | 0.818 ± 0.028 | 0.642 ± 0.032 | 0.643 ± 0.030 | 0.572 ± 0.039 |
| MobileNetV2 | 0.568 ± 0.011 | 0.858 ± 0.011 | 0.666 ± 0.011 | 0.652 ± 0.017 | 0.627 ± 0.043 |
| DINOv2 probe | 0.624 ± 0.020 | 0.873 ± 0.003 | 0.700 ± 0.018 | 0.702 ± 0.011 | 0.708 ± 0.012 |

## Leakage (seed 42 test): image vs lesion bal acc, diff, AUROC image vs lesion
EffNet 0.572/0.568 +0.3 0.844/0.836; MobileNet 0.570/0.556 +1.4 0.849/0.861; ResNet 0.670/0.673 -0.3 0.902/0.913; Xception 0.675/0.647 +2.8 0.854/0.889. Range -0.3..+2.8, mean +1.0.

## alpha .05 (miss / mel miss / referral / breaks / domain AUC)
HAM internal (0.52): SRC .048/.049/.653/27.8%; WCP .041/.039/.669/44.4%
BCN (0.94): SRC .172±.096/.157±.085/.764±.123/94.4%; WCP .046±.040/.052±.046/.909±.079/38.9%; QTC .247±.116/.231±.107/.665±.151/94.4%; QTC-LS .251±.155/.236±.138/.666±.181/94.4%; OR20 .063; OR50 .026±.023/.023±.021/.968±.029/16.7%
MILK (0.92): SRC .139±.065/.143±.035/.829±.070/94.4%; WCP .076±.056/.134±.109/.891±.077/72.2%; QTC .289±.141/.328±.179/.665±.151/94.4%; QTC-LS .229±.148/.232±.140/.733±.159/88.9%; OR50 .038±.025/.039±.032/.949±.031/44.4%
PAD (0.96): SRC .207±.076/.277±.144/.757±.089/100%; WCP .015±.016/.053±.070/.978±.026/0%; QTC .288±.132/.399±.202/.665±.151/94.4%; QTC-LS .297±.145/.409±.214/.655±.160/94.4%; OR50 .042±.026/.040±.038/.950±.033/27.8%
MILK subgroups SRC/WCP miss (WCP referral): head_neck .129/.065 (.912); lower_ext .167/.045 (.921); trunk .117/.058 (.897); upper_ext .158/.038 (.941)

## alpha .10 key
BCN SRC .285 (100%), WCP .120 ref .809 (38.9%), QTC .403; MILK SRC .226 (100%), WCP .186 ref .770 (72.2%), QTC .477, QTC-LS .368; PAD SRC .319 (100%), WCP .038 ref .948 (11.1%), QTC .474

## Paired tests (54 pairs external), Holm
.05 SRC vs WCP: Δmiss +0.116, Δref -0.136, only SRC breaks/only WCP 32/0; p 2.8e-9 / 2.7e-8 / 2.8e-9
SRC vs QTC Δmiss -0.11 p 2.0e-7; SRC vs QTC-LS -0.083 p 8.4e-6; QTC vs LS p .99
.10 SRC vs WCP Δmiss +0.179 Δref -0.216 32/0 p 2.0e-7 / 1.8e-7 / 2.8e-9

## Key external alpha .05
SRC miss .173 (max .207), breaks 96.3%, referral .783, mel .192
WCP miss .046 (max .076), breaks 37%, referral .926, mel .080
QTC .275 breaks 94.4% ref .665; QTC-LS .259 breaks 92.6% ref .685

## Per model (alpha .05; mean over seeds & external sites): SRC miss / WCP miss / SRC ref / WCP ref / SRC breaks / WCP breaks / OR50 miss
ConvNeXt-T .101/.061/.861/.874/100%/55.6%/.032
DINOv2 .160/.026/.783/.956/100%/33.3%/.040
EfficientNet-B3 .138/.026/.855/.968/77.8%/22.2%/.032
MobileNetV2 .158/.028/.809/.957/100%/11.1%/.039
ResNet-50 .202/.055/.732/.912/100%/55.6%/.036
Xception .278/.077/.662/.889/100%/44.4%/.033
