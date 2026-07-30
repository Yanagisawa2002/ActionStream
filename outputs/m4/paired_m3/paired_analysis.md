# ActionStream M4 Section A: paired M3 evidence

The episode is the statistical unit; the 29,073 actions are not independent samples. Pairs share task, episode/initial-state index, and seed. Differences below are **estimate minus reference**.

Bootstrap: 10,000 paired resamples, fixed seed `20260730`, percentile 95% intervals.

Observation-to-delivery is the existing episode p50 summarized over episodes. Discontinuity is the existing episode mean adjacent-command 7D L2.

## Raw episode summaries

| mode | delay | scope | n | success mean/median | steps mean/median | wall s mean/median | delivery s mean/median | discontinuity mean/median |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| async_aligned | 0 | all | 30 | 1.0000/1.0000 | 136.6000/134.5000 | 9.2349/9.0702 | 0.1309/0.1307 | 0.1584/0.1428 |
| async_aligned | 0 | task 0 | 10 | 1.0000/1.0000 | 151.4000/151.5000 | 10.2811/9.9139 | 0.1295/0.1283 | 0.1501/0.1584 |
| async_aligned | 0 | task 1 | 10 | 1.0000/1.0000 | 133.6000/131.5000 | 8.8266/8.7748 | 0.1307/0.1293 | 0.1446/0.1443 |
| async_aligned | 0 | task 2 | 10 | 1.0000/1.0000 | 124.8000/123.5000 | 8.5969/8.5137 | 0.1324/0.1336 | 0.1805/0.1354 |
| async_aligned | 200 | all | 30 | 1.0000/1.0000 | 142.7333/130.5000 | 10.0753/8.8973 | 0.4569/0.4578 | 0.1827/0.1506 |
| async_aligned | 200 | task 0 | 10 | 1.0000/1.0000 | 144.9000/147.0000 | 11.0646/9.7623 | 0.4576/0.4578 | 0.1626/0.1426 |
| async_aligned | 200 | task 1 | 10 | 1.0000/1.0000 | 164.6000/130.5000 | 10.7146/8.8827 | 0.4565/0.4597 | 0.1928/0.1665 |
| async_aligned | 200 | task 2 | 10 | 1.0000/1.0000 | 118.7000/118.5000 | 8.4465/8.4382 | 0.4565/0.4564 | 0.1927/0.1416 |
| async_naive | 0 | all | 30 | 1.0000/1.0000 | 189.7000/185.5000 | 11.9468/11.6113 | 0.1281/0.1280 | 0.2102/0.1830 |
| async_naive | 0 | task 0 | 10 | 1.0000/1.0000 | 213.1000/213.5000 | 13.3831/13.2781 | 0.1298/0.1296 | 0.1960/0.1776 |
| async_naive | 0 | task 1 | 10 | 1.0000/1.0000 | 185.3000/184.0000 | 11.4835/11.2526 | 0.1272/0.1270 | 0.2061/0.1784 |
| async_naive | 0 | task 2 | 10 | 1.0000/1.0000 | 170.7000/170.0000 | 10.9737/10.8941 | 0.1272/0.1275 | 0.2286/0.1972 |
| async_naive | 200 | all | 30 | 1.0000/1.0000 | 246.2667/238.0000 | 15.2171/14.9321 | 0.4516/0.4558 | 0.2748/0.2694 |
| async_naive | 200 | task 0 | 10 | 1.0000/1.0000 | 275.4000/276.0000 | 17.0287/17.0901 | 0.4553/0.4583 | 0.2195/0.2239 |
| async_naive | 200 | task 1 | 10 | 1.0000/1.0000 | 242.9000/238.0000 | 14.8034/14.6237 | 0.4442/0.4527 | 0.2693/0.2772 |
| async_naive | 200 | task 2 | 10 | 1.0000/1.0000 | 220.5000/217.0000 | 13.8191/13.6447 | 0.4553/0.4562 | 0.3355/0.3019 |
| blocking evaluator baseline | 0 | all | 30 | 1.0000/1.0000 | 126.9000/126.0000 | 8.8309/8.6525 | 0.1510/0.1567 | 0.1147/0.1320 |
| blocking evaluator baseline | 0 | task 0 | 10 | 1.0000/1.0000 | 141.7000/141.5000 | 9.7516/9.5821 | 0.1448/0.1483 | 0.1023/0.1023 |
| blocking evaluator baseline | 0 | task 1 | 10 | 1.0000/1.0000 | 124.4000/126.0000 | 8.6431/8.6525 | 0.1432/0.1521 | 0.0974/0.0934 |
| blocking evaluator baseline | 0 | task 2 | 10 | 1.0000/1.0000 | 114.6000/114.5000 | 8.0979/8.0105 | 0.1651/0.1612 | 0.1446/0.1410 |
| blocking evaluator baseline | 200 | all | 30 | 1.0000/1.0000 | 126.9000/126.0000 | 10.4347/10.1402 | 0.4897/0.5046 | 0.1147/0.1320 |
| blocking evaluator baseline | 200 | task 0 | 10 | 1.0000/1.0000 | 141.7000/141.5000 | 11.7712/11.5694 | 0.5093/0.5079 | 0.1023/0.1023 |
| blocking evaluator baseline | 200 | task 1 | 10 | 1.0000/1.0000 | 124.4000/126.0000 | 10.1890/10.1402 | 0.4989/0.5093 | 0.0974/0.0934 |
| blocking evaluator baseline | 200 | task 2 | 10 | 1.0000/1.0000 | 114.6000/114.5000 | 9.3440/9.4012 | 0.4610/0.4568 | 0.1446/0.1410 |

## Paired effects

Relative effects are means of paired episode-wise relative differences.

| comparison | scope | metric | n | reference mean/median | estimate mean/median | mean difference [95% CI] | relative difference [95% CI] |
|---|---|---|---:|---:|---:|---:|---:|
| async_aligned/0 minus async_naive/0 | all | environment steps | 30 | 189.7000/185.5000 | 136.6000/134.5000 | -53.1000 [-56.3667, -50.0667] | -27.89% [-28.81, -26.97]% |
| async_aligned/0 minus async_naive/0 | all | wall-clock episode seconds | 30 | 11.9468/11.6113 | 9.2349/9.0702 | -2.7119 [-2.9207, -2.5108] | -22.64% [-23.95, -21.28]% |
| async_aligned/0 minus async_naive/0 | all | episode p50 observation-to-delivery seconds | 30 | 0.1281/0.1280 | 0.1309/0.1307 | 0.0028 [0.0007, 0.0049] | 2.23% [0.57, 3.91]% |
| async_aligned/0 minus async_naive/0 | all | episode mean adjacent-action 7D L2 | 30 | 0.2102/0.1830 | 0.1584/0.1428 | -0.0518 [-0.0889, -0.0154] | -6.02% [-27.03, 18.25]% |
| async_aligned/0 minus async_naive/0 | all | binary episode success | 30 | 1.0000/1.0000 | 1.0000/1.0000 | 0.0000 [0.0000, 0.0000] | 0.00% [0.00, 0.00]% |
| async_aligned/0 minus async_naive/0 | task 0 | environment steps | 10 | 213.1000/213.5000 | 151.4000/151.5000 | -61.7000 [-67.0000, -55.6000] | -28.83% [-30.75, -26.55]% |
| async_aligned/0 minus async_naive/0 | task 0 | wall-clock episode seconds | 10 | 13.3831/13.2781 | 10.2811/9.9139 | -3.1021 [-3.4999, -2.6153] | -23.16% [-26.12, -19.61]% |
| async_aligned/0 minus async_naive/0 | task 0 | episode p50 observation-to-delivery seconds | 10 | 0.1298/0.1296 | 0.1295/0.1283 | -0.0003 [-0.0032, 0.0027] | -0.18% [-2.40, 2.24]% |
| async_aligned/0 minus async_naive/0 | task 0 | episode mean adjacent-action 7D L2 | 10 | 0.1960/0.1776 | 0.1501/0.1584 | -0.0458 [-0.0989, 0.0101] | 2.10% [-39.65, 60.22]% |
| async_aligned/0 minus async_naive/0 | task 0 | binary episode success | 10 | 1.0000/1.0000 | 1.0000/1.0000 | 0.0000 [0.0000, 0.0000] | 0.00% [0.00, 0.00]% |
| async_aligned/0 minus async_naive/0 | task 1 | environment steps | 10 | 185.3000/184.0000 | 133.6000/131.5000 | -51.7000 [-53.1000, -50.3000] | -27.93% [-28.91, -26.94]% |
| async_aligned/0 minus async_naive/0 | task 1 | wall-clock episode seconds | 10 | 11.4835/11.2526 | 8.8266/8.7748 | -2.6569 [-2.8861, -2.4448] | -23.11% [-24.76, -21.48]% |
| async_aligned/0 minus async_naive/0 | task 1 | episode p50 observation-to-delivery seconds | 10 | 0.1272/0.1270 | 0.1307/0.1293 | 0.0034 [0.0010, 0.0060] | 2.70% [0.81, 4.76]% |
| async_aligned/0 minus async_naive/0 | task 1 | episode mean adjacent-action 7D L2 | 10 | 0.2061/0.1784 | 0.1446/0.1443 | -0.0615 [-0.1252, 0.0021] | -15.08% [-41.87, 16.87]% |
| async_aligned/0 minus async_naive/0 | task 1 | binary episode success | 10 | 1.0000/1.0000 | 1.0000/1.0000 | 0.0000 [0.0000, 0.0000] | 0.00% [0.00, 0.00]% |
| async_aligned/0 minus async_naive/0 | task 2 | environment steps | 10 | 170.7000/170.0000 | 124.8000/123.5000 | -45.9000 [-48.1000, -43.9000] | -26.90% [-28.07, -25.81]% |
| async_aligned/0 minus async_naive/0 | task 2 | wall-clock episode seconds | 10 | 10.9737/10.8941 | 8.5969/8.5137 | -2.3768 [-2.5181, -2.2072] | -21.65% [-22.70, -20.18]% |
| async_aligned/0 minus async_naive/0 | task 2 | episode p50 observation-to-delivery seconds | 10 | 0.1272/0.1275 | 0.1324/0.1336 | 0.0052 [0.0008, 0.0093] | 4.18% [0.66, 7.46]% |
| async_aligned/0 minus async_naive/0 | task 2 | episode mean adjacent-action 7D L2 | 10 | 0.2286/0.1972 | 0.1805/0.1354 | -0.0481 [-0.1221, 0.0255] | -5.07% [-33.62, 30.91]% |
| async_aligned/0 minus async_naive/0 | task 2 | binary episode success | 10 | 1.0000/1.0000 | 1.0000/1.0000 | 0.0000 [0.0000, 0.0000] | 0.00% [0.00, 0.00]% |
| async_aligned/0 minus blocking evaluator baseline/0 | all | environment steps | 30 | 126.9000/126.0000 | 136.6000/134.5000 | 9.7000 [7.9333, 11.6667] | 7.71% [6.31, 9.24]% |
| async_aligned/0 minus blocking evaluator baseline/0 | all | wall-clock episode seconds | 30 | 8.8309/8.6525 | 9.2349/9.0702 | 0.4040 [0.2379, 0.5807] | 4.63% [2.77, 6.60]% |
| async_aligned/0 minus blocking evaluator baseline/0 | all | episode p50 observation-to-delivery seconds | 30 | 0.1510/0.1567 | 0.1309/0.1307 | -0.0202 [-0.0290, -0.0111] | -10.85% [-16.60, -4.69]% |
| async_aligned/0 minus blocking evaluator baseline/0 | all | episode mean adjacent-action 7D L2 | 30 | 0.1147/0.1320 | 0.1584/0.1428 | 0.0437 [0.0120, 0.0752] | 81.37% [39.04, 126.80]% |
| async_aligned/0 minus blocking evaluator baseline/0 | all | binary episode success | 30 | 1.0000/1.0000 | 1.0000/1.0000 | 0.0000 [0.0000, 0.0000] | 0.00% [0.00, 0.00]% |
| async_aligned/0 minus blocking evaluator baseline/0 | task 0 | environment steps | 10 | 141.7000/141.5000 | 151.4000/151.5000 | 9.7000 [5.9000, 14.3000] | 6.88% [4.13, 10.18]% |
| async_aligned/0 minus blocking evaluator baseline/0 | task 0 | wall-clock episode seconds | 10 | 9.7516/9.5821 | 10.2811/9.9139 | 0.5295 [0.1795, 0.9115] | 5.52% [1.81, 9.59]% |
| async_aligned/0 minus blocking evaluator baseline/0 | task 0 | episode p50 observation-to-delivery seconds | 10 | 0.1448/0.1483 | 0.1295/0.1283 | -0.0153 [-0.0331, 0.0018] | -7.06% [-18.56, 5.16]% |
| async_aligned/0 minus blocking evaluator baseline/0 | task 0 | episode mean adjacent-action 7D L2 | 10 | 0.1023/0.1023 | 0.1501/0.1584 | 0.0479 [-0.0104, 0.1024] | 78.59% [6.98, 161.93]% |
| async_aligned/0 minus blocking evaluator baseline/0 | task 0 | binary episode success | 10 | 1.0000/1.0000 | 1.0000/1.0000 | 0.0000 [0.0000, 0.0000] | 0.00% [0.00, 0.00]% |
| async_aligned/0 minus blocking evaluator baseline/0 | task 1 | environment steps | 10 | 124.4000/126.0000 | 133.6000/131.5000 | 9.2000 [6.5000, 11.9000] | 7.41% [5.27, 9.60]% |
| async_aligned/0 minus blocking evaluator baseline/0 | task 1 | wall-clock episode seconds | 10 | 8.6431/8.6525 | 8.8266/8.7748 | 0.1835 [-0.0645, 0.4414] | 2.24% [-0.65, 5.25]% |
| async_aligned/0 minus blocking evaluator baseline/0 | task 1 | episode p50 observation-to-delivery seconds | 10 | 0.1432/0.1521 | 0.1307/0.1293 | -0.0125 [-0.0278, 0.0032] | -6.10% [-16.48, 4.81]% |
| async_aligned/0 minus blocking evaluator baseline/0 | task 1 | episode mean adjacent-action 7D L2 | 10 | 0.0974/0.0934 | 0.1446/0.1443 | 0.0472 [0.0091, 0.0893] | 95.77% [20.15, 183.86]% |
| async_aligned/0 minus blocking evaluator baseline/0 | task 1 | binary episode success | 10 | 1.0000/1.0000 | 1.0000/1.0000 | 0.0000 [0.0000, 0.0000] | 0.00% [0.00, 0.00]% |
| async_aligned/0 minus blocking evaluator baseline/0 | task 2 | environment steps | 10 | 114.6000/114.5000 | 124.8000/123.5000 | 10.2000 [7.8000, 13.2000] | 8.85% [6.88, 11.37]% |
| async_aligned/0 minus blocking evaluator baseline/0 | task 2 | wall-clock episode seconds | 10 | 8.0979/8.0105 | 8.5969/8.5137 | 0.4990 [0.3105, 0.7144] | 6.14% [3.84, 8.75]% |
| async_aligned/0 minus blocking evaluator baseline/0 | task 2 | episode p50 observation-to-delivery seconds | 10 | 0.1651/0.1612 | 0.1324/0.1336 | -0.0327 [-0.0411, -0.0247] | -19.38% [-23.53, -15.27]% |
| async_aligned/0 minus blocking evaluator baseline/0 | task 2 | episode mean adjacent-action 7D L2 | 10 | 0.1446/0.1410 | 0.1805/0.1354 | 0.0359 [-0.0271, 0.1001] | 69.74% [6.75, 141.71]% |
| async_aligned/0 minus blocking evaluator baseline/0 | task 2 | binary episode success | 10 | 1.0000/1.0000 | 1.0000/1.0000 | 0.0000 [0.0000, 0.0000] | 0.00% [0.00, 0.00]% |
| async_aligned/200 minus async_naive/200 | all | environment steps | 30 | 246.2667/238.0000 | 142.7333/130.5000 | -103.5333 [-118.8333, -79.9992] | -41.77% [-47.20, -32.33]% |
| async_aligned/200 minus async_naive/200 | all | wall-clock episode seconds | 30 | 15.2171/14.9321 | 10.0753/8.8973 | -5.1418 [-6.2325, -3.6789] | -33.89% [-40.61, -24.53]% |
| async_aligned/200 minus async_naive/200 | all | episode p50 observation-to-delivery seconds | 30 | 0.4516/0.4558 | 0.4569/0.4578 | 0.0053 [-0.0000, 0.0133] | 1.39% [0.00, 3.57]% |
| async_aligned/200 minus async_naive/200 | all | episode mean adjacent-action 7D L2 | 30 | 0.2748/0.2694 | 0.1827/0.1506 | -0.0921 [-0.1289, -0.0539] | -19.39% [-40.14, 7.21]% |
| async_aligned/200 minus async_naive/200 | all | binary episode success | 30 | 1.0000/1.0000 | 1.0000/1.0000 | 0.0000 [0.0000, 0.0000] | 0.00% [0.00, 0.00]% |
| async_aligned/200 minus async_naive/200 | task 0 | environment steps | 10 | 275.4000/276.0000 | 144.9000/147.0000 | -130.5000 [-139.6000, -119.3000] | -47.19% [-49.28, -44.61]% |
| async_aligned/200 minus async_naive/200 | task 0 | wall-clock episode seconds | 10 | 17.0287/17.0901 | 11.0646/9.7623 | -5.9641 [-7.5213, -3.3111] | -35.36% [-43.85, -20.23]% |
| async_aligned/200 minus async_naive/200 | task 0 | episode p50 observation-to-delivery seconds | 10 | 0.4553/0.4583 | 0.4576/0.4578 | 0.0024 [-0.0020, 0.0074] | 0.55% [-0.41, 1.66]% |
| async_aligned/200 minus async_naive/200 | task 0 | episode mean adjacent-action 7D L2 | 10 | 0.2195/0.2239 | 0.1626/0.1426 | -0.0569 [-0.1168, 0.0098] | -13.91% [-51.13, 33.16]% |
| async_aligned/200 minus async_naive/200 | task 0 | binary episode success | 10 | 1.0000/1.0000 | 1.0000/1.0000 | 0.0000 [0.0000, 0.0000] | 0.00% [0.00, 0.00]% |
| async_aligned/200 minus async_naive/200 | task 1 | environment steps | 10 | 242.9000/238.0000 | 164.6000/130.5000 | -78.3000 [-115.4000, -16.5000] | -32.03% [-47.40, -6.49]% |
| async_aligned/200 minus async_naive/200 | task 1 | wall-clock episode seconds | 10 | 14.8034/14.6237 | 10.7146/8.8827 | -4.0888 [-6.1356, -0.9021] | -27.51% [-41.20, -6.17]% |
| async_aligned/200 minus async_naive/200 | task 1 | episode p50 observation-to-delivery seconds | 10 | 0.4442/0.4527 | 0.4565/0.4597 | 0.0123 [-0.0015, 0.0344] | 3.33% [-0.33, 9.42]% |
| async_aligned/200 minus async_naive/200 | task 1 | episode mean adjacent-action 7D L2 | 10 | 0.2693/0.2772 | 0.1928/0.1665 | -0.0765 [-0.1337, -0.0127] | -4.43% [-44.63, 57.52]% |
| async_aligned/200 minus async_naive/200 | task 1 | binary episode success | 10 | 1.0000/1.0000 | 1.0000/1.0000 | 0.0000 [0.0000, 0.0000] | 0.00% [0.00, 0.00]% |
| async_aligned/200 minus async_naive/200 | task 2 | environment steps | 10 | 220.5000/217.0000 | 118.7000/118.5000 | -101.8000 [-108.8000, -96.2000] | -46.08% [-47.65, -44.61]% |
| async_aligned/200 minus async_naive/200 | task 2 | wall-clock episode seconds | 10 | 13.8191/13.6447 | 8.4465/8.4382 | -5.3726 [-5.7868, -5.0591] | -38.79% [-40.54, -37.30]% |
| async_aligned/200 minus async_naive/200 | task 2 | episode p50 observation-to-delivery seconds | 10 | 0.4553/0.4562 | 0.4565/0.4564 | 0.0013 [-0.0027, 0.0050] | 0.29% [-0.58, 1.12]% |
| async_aligned/200 minus async_naive/200 | task 2 | episode mean adjacent-action 7D L2 | 10 | 0.3355/0.3019 | 0.1927/0.1416 | -0.1429 [-0.2040, -0.0825] | -39.84% [-51.33, -26.55]% |
| async_aligned/200 minus async_naive/200 | task 2 | binary episode success | 10 | 1.0000/1.0000 | 1.0000/1.0000 | 0.0000 [0.0000, 0.0000] | 0.00% [0.00, 0.00]% |
| async_aligned/200 minus blocking evaluator baseline/200 | all | environment steps | 30 | 126.9000/126.0000 | 142.7333/130.5000 | 15.8333 [3.1000, 37.9675] | 12.80% [2.51, 30.75]% |
| async_aligned/200 minus blocking evaluator baseline/200 | all | wall-clock episode seconds | 30 | 10.4347/10.1402 | 10.0753/8.8973 | -0.3595 [-1.4012, 1.0601] | -3.64% [-13.12, 9.68]% |
| async_aligned/200 minus blocking evaluator baseline/200 | all | episode p50 observation-to-delivery seconds | 30 | 0.4897/0.5046 | 0.4569/0.4578 | -0.0329 [-0.0483, -0.0159] | -5.85% [-9.04, -2.25]% |
| async_aligned/200 minus blocking evaluator baseline/200 | all | episode mean adjacent-action 7D L2 | 30 | 0.1147/0.1320 | 0.1827/0.1506 | 0.0679 [0.0347, 0.1013] | 96.45% [52.05, 144.53]% |
| async_aligned/200 minus blocking evaluator baseline/200 | all | binary episode success | 30 | 1.0000/1.0000 | 1.0000/1.0000 | 0.0000 [0.0000, 0.0000] | 0.00% [0.00, 0.00]% |
| async_aligned/200 minus blocking evaluator baseline/200 | task 0 | environment steps | 10 | 141.7000/141.5000 | 144.9000/147.0000 | 3.2000 [-0.8000, 7.6000] | 2.29% [-0.57, 5.42]% |
| async_aligned/200 minus blocking evaluator baseline/200 | task 0 | wall-clock episode seconds | 10 | 11.7712/11.5694 | 11.0646/9.7623 | -0.7066 [-2.1545, 1.8872] | -6.85% [-18.25, 13.64]% |
| async_aligned/200 minus blocking evaluator baseline/200 | task 0 | episode p50 observation-to-delivery seconds | 10 | 0.5093/0.5079 | 0.4576/0.4578 | -0.0517 [-0.0732, -0.0296] | -9.70% [-13.50, -5.49]% |
| async_aligned/200 minus blocking evaluator baseline/200 | task 0 | episode mean adjacent-action 7D L2 | 10 | 0.1023/0.1023 | 0.1626/0.1426 | 0.0603 [0.0141, 0.1081] | 55.28% [12.83, 102.12]% |
| async_aligned/200 minus blocking evaluator baseline/200 | task 0 | binary episode success | 10 | 1.0000/1.0000 | 1.0000/1.0000 | 0.0000 [0.0000, 0.0000] | 0.00% [0.00, 0.00]% |
| async_aligned/200 minus blocking evaluator baseline/200 | task 1 | environment steps | 10 | 124.4000/126.0000 | 164.6000/130.5000 | 40.2000 [3.3000, 100.5075] | 32.53% [2.66, 81.90]% |
| async_aligned/200 minus blocking evaluator baseline/200 | task 1 | wall-clock episode seconds | 10 | 10.1890/10.1402 | 10.7146/8.8827 | 0.5256 [-1.4989, 3.7175] | 5.46% [-14.46, 37.04]% |
| async_aligned/200 minus blocking evaluator baseline/200 | task 1 | episode p50 observation-to-delivery seconds | 10 | 0.4989/0.5093 | 0.4565/0.4597 | -0.0424 [-0.0617, -0.0156] | -7.92% [-11.91, -2.07]% |
| async_aligned/200 minus blocking evaluator baseline/200 | task 1 | episode mean adjacent-action 7D L2 | 10 | 0.0974/0.0934 | 0.1928/0.1665 | 0.0954 [0.0389, 0.1586] | 153.96% [58.33, 256.20]% |
| async_aligned/200 minus blocking evaluator baseline/200 | task 1 | binary episode success | 10 | 1.0000/1.0000 | 1.0000/1.0000 | 0.0000 [0.0000, 0.0000] | 0.00% [0.00, 0.00]% |
| async_aligned/200 minus blocking evaluator baseline/200 | task 2 | environment steps | 10 | 114.6000/114.5000 | 118.7000/118.5000 | 4.1000 [1.9000, 6.8000] | 3.59% [1.67, 5.98]% |
| async_aligned/200 minus blocking evaluator baseline/200 | task 2 | wall-clock episode seconds | 10 | 9.3440/9.4012 | 8.4465/8.4382 | -0.8975 [-1.1174, -0.6400] | -9.54% [-11.87, -6.84]% |
| async_aligned/200 minus blocking evaluator baseline/200 | task 2 | episode p50 observation-to-delivery seconds | 10 | 0.4610/0.4568 | 0.4565/0.4564 | -0.0045 [-0.0331, 0.0241] | 0.07% [-6.03, 6.34]% |
| async_aligned/200 minus blocking evaluator baseline/200 | task 2 | episode mean adjacent-action 7D L2 | 10 | 0.1446/0.1410 | 0.1927/0.1416 | 0.0481 [-0.0125, 0.1114] | 80.11% [9.70, 160.81]% |
| async_aligned/200 minus blocking evaluator baseline/200 | task 2 | binary episode success | 10 | 1.0000/1.0000 | 1.0000/1.0000 | 0.0000 [0.0000, 0.0000] | 0.00% [0.00, 0.00]% |
| blocking evaluator baseline/200 minus blocking evaluator baseline/0 | all | environment steps | 30 | 126.9000/126.0000 | 126.9000/126.0000 | 0.0000 [0.0000, 0.0000] | 0.00% [0.00, 0.00]% |
| blocking evaluator baseline/200 minus blocking evaluator baseline/0 | all | wall-clock episode seconds | 30 | 8.8309/8.6525 | 10.4347/10.1402 | 1.6039 [1.4265, 1.8069] | 18.13% [16.28, 20.23]% |
| blocking evaluator baseline/200 minus blocking evaluator baseline/0 | all | episode p50 observation-to-delivery seconds | 30 | 0.1510/0.1567 | 0.4897/0.5046 | 0.3387 [0.3180, 0.3585] | 234.98% [208.92, 262.69]% |
| blocking evaluator baseline/200 minus blocking evaluator baseline/0 | all | episode mean adjacent-action 7D L2 | 30 | 0.1147/0.1320 | 0.1147/0.1320 | 0.0000 [0.0000, 0.0000] | 0.00% [0.00, 0.00]% |
| blocking evaluator baseline/200 minus blocking evaluator baseline/0 | all | binary episode success | 30 | 1.0000/1.0000 | 1.0000/1.0000 | 0.0000 [0.0000, 0.0000] | 0.00% [0.00, 0.00]% |
| blocking evaluator baseline/200 minus blocking evaluator baseline/0 | task 0 | environment steps | 10 | 141.7000/141.5000 | 141.7000/141.5000 | 0.0000 [0.0000, 0.0000] | 0.00% [0.00, 0.00]% |
| blocking evaluator baseline/200 minus blocking evaluator baseline/0 | task 0 | wall-clock episode seconds | 10 | 9.7516/9.5821 | 11.7712/11.5694 | 2.0196 [1.6902, 2.4452] | 21.07% [17.11, 26.03]% |
| blocking evaluator baseline/200 minus blocking evaluator baseline/0 | task 0 | episode p50 observation-to-delivery seconds | 10 | 0.1448/0.1483 | 0.5093/0.5079 | 0.3645 [0.3350, 0.3962] | 266.67% [216.81, 323.42]% |
| blocking evaluator baseline/200 minus blocking evaluator baseline/0 | task 0 | episode mean adjacent-action 7D L2 | 10 | 0.1023/0.1023 | 0.1023/0.1023 | 0.0000 [0.0000, 0.0000] | 0.00% [0.00, 0.00]% |
| blocking evaluator baseline/200 minus blocking evaluator baseline/0 | task 0 | binary episode success | 10 | 1.0000/1.0000 | 1.0000/1.0000 | 0.0000 [0.0000, 0.0000] | 0.00% [0.00, 0.00]% |
| blocking evaluator baseline/200 minus blocking evaluator baseline/0 | task 1 | environment steps | 10 | 124.4000/126.0000 | 124.4000/126.0000 | 0.0000 [0.0000, 0.0000] | 0.00% [0.00, 0.00]% |
| blocking evaluator baseline/200 minus blocking evaluator baseline/0 | task 1 | wall-clock episode seconds | 10 | 8.6431/8.6525 | 10.1890/10.1402 | 1.5459 [1.3202, 1.7952] | 17.93% [15.25, 20.80]% |
| blocking evaluator baseline/200 minus blocking evaluator baseline/0 | task 1 | episode p50 observation-to-delivery seconds | 10 | 0.1432/0.1521 | 0.4989/0.5093 | 0.3557 [0.3285, 0.3750] | 256.72% [219.99, 293.64]% |
| blocking evaluator baseline/200 minus blocking evaluator baseline/0 | task 1 | episode mean adjacent-action 7D L2 | 10 | 0.0974/0.0934 | 0.0974/0.0934 | 0.0000 [0.0000, 0.0000] | 0.00% [0.00, 0.00]% |
| blocking evaluator baseline/200 minus blocking evaluator baseline/0 | task 1 | binary episode success | 10 | 1.0000/1.0000 | 1.0000/1.0000 | 0.0000 [0.0000, 0.0000] | 0.00% [0.00, 0.00]% |
| blocking evaluator baseline/200 minus blocking evaluator baseline/0 | task 2 | environment steps | 10 | 114.6000/114.5000 | 114.6000/114.5000 | 0.0000 [0.0000, 0.0000] | 0.00% [0.00, 0.00]% |
| blocking evaluator baseline/200 minus blocking evaluator baseline/0 | task 2 | wall-clock episode seconds | 10 | 8.0979/8.0105 | 9.3440/9.4012 | 1.2462 [1.1365, 1.3535] | 15.40% [14.02, 16.79]% |
| blocking evaluator baseline/200 minus blocking evaluator baseline/0 | task 2 | episode p50 observation-to-delivery seconds | 10 | 0.1651/0.1612 | 0.4610/0.4568 | 0.2959 [0.2649, 0.3297] | 181.55% [157.13, 209.04]% |
| blocking evaluator baseline/200 minus blocking evaluator baseline/0 | task 2 | episode mean adjacent-action 7D L2 | 10 | 0.1446/0.1410 | 0.1446/0.1410 | 0.0000 [0.0000, 0.0000] | 0.00% [0.00, 0.00]% |
| blocking evaluator baseline/200 minus blocking evaluator baseline/0 | task 2 | binary episode success | 10 | 1.0000/1.0000 | 1.0000/1.0000 | 0.0000 [0.0000, 0.0000] | 0.00% [0.00, 0.00]% |
| async_naive/200 minus async_naive/0 | all | environment steps | 30 | 189.7000/185.5000 | 246.2667/238.0000 | 56.5667 [53.3667, 59.6333] | 29.81% [28.55, 31.01]% |
| async_naive/200 minus async_naive/0 | all | wall-clock episode seconds | 30 | 11.9468/11.6113 | 15.2171/14.9321 | 3.2703 [3.0886, 3.4399] | 27.41% [26.11, 28.58]% |
| async_naive/200 minus async_naive/0 | all | episode p50 observation-to-delivery seconds | 30 | 0.1281/0.1280 | 0.4516/0.4558 | 0.3235 [0.3162, 0.3283] | 252.74% [246.75, 257.90]% |
| async_naive/200 minus async_naive/0 | all | episode mean adjacent-action 7D L2 | 30 | 0.2102/0.1830 | 0.2748/0.2694 | 0.0645 [0.0243, 0.1053] | 72.29% [25.46, 130.35]% |
| async_naive/200 minus async_naive/0 | all | binary episode success | 30 | 1.0000/1.0000 | 1.0000/1.0000 | 0.0000 [0.0000, 0.0000] | 0.00% [0.00, 0.00]% |
| async_naive/200 minus async_naive/0 | task 0 | environment steps | 10 | 213.1000/213.5000 | 275.4000/276.0000 | 62.3000 [56.6000, 67.3000] | 29.21% [26.83, 31.50]% |
| async_naive/200 minus async_naive/0 | task 0 | wall-clock episode seconds | 10 | 13.3831/13.2781 | 17.0287/17.0901 | 3.6456 [3.3901, 3.8676] | 27.33% [25.13, 29.31]% |
| async_naive/200 minus async_naive/0 | task 0 | episode p50 observation-to-delivery seconds | 10 | 0.1298/0.1296 | 0.4553/0.4583 | 0.3254 [0.3202, 0.3301] | 250.91% [244.38, 257.87]% |
| async_naive/200 minus async_naive/0 | task 0 | episode mean adjacent-action 7D L2 | 10 | 0.1960/0.1776 | 0.2195/0.2239 | 0.0235 [-0.0421, 0.0897] | 100.79% [-13.08, 261.05]% |
| async_naive/200 minus async_naive/0 | task 0 | binary episode success | 10 | 1.0000/1.0000 | 1.0000/1.0000 | 0.0000 [0.0000, 0.0000] | 0.00% [0.00, 0.00]% |
| async_naive/200 minus async_naive/0 | task 1 | environment steps | 10 | 185.3000/184.0000 | 242.9000/238.0000 | 57.6000 [54.4000, 61.0025] | 31.06% [29.64, 32.66]% |
| async_naive/200 minus async_naive/0 | task 1 | wall-clock episode seconds | 10 | 11.4835/11.2526 | 14.8034/14.6237 | 3.3199 [3.1488, 3.4873] | 28.97% [27.23, 30.60]% |
| async_naive/200 minus async_naive/0 | task 1 | episode p50 observation-to-delivery seconds | 10 | 0.1272/0.1270 | 0.4442/0.4527 | 0.3169 [0.2974, 0.3283] | 249.17% [233.50, 260.29]% |
| async_naive/200 minus async_naive/0 | task 1 | episode mean adjacent-action 7D L2 | 10 | 0.2061/0.1784 | 0.2693/0.2772 | 0.0632 [-0.0128, 0.1467] | 54.59% [-0.36, 119.83]% |
| async_naive/200 minus async_naive/0 | task 1 | binary episode success | 10 | 1.0000/1.0000 | 1.0000/1.0000 | 0.0000 [0.0000, 0.0000] | 0.00% [0.00, 0.00]% |
| async_naive/200 minus async_naive/0 | task 2 | environment steps | 10 | 170.7000/170.0000 | 220.5000/217.0000 | 49.8000 [45.7000, 54.0000] | 29.16% [26.92, 31.31]% |
| async_naive/200 minus async_naive/0 | task 2 | wall-clock episode seconds | 10 | 10.9737/10.8941 | 13.8191/13.6447 | 2.8455 [2.5993, 3.0726] | 25.94% [23.66, 27.76]% |
| async_naive/200 minus async_naive/0 | task 2 | episode p50 observation-to-delivery seconds | 10 | 0.1272/0.1275 | 0.4553/0.4562 | 0.3280 [0.3247, 0.3308] | 258.13% [252.22, 263.79]% |
| async_naive/200 minus async_naive/0 | task 2 | episode mean adjacent-action 7D L2 | 10 | 0.2286/0.1972 | 0.3355/0.3019 | 0.1069 [0.0603, 0.1644] | 61.49% [34.24, 93.68]% |
| async_naive/200 minus async_naive/0 | task 2 | binary episode success | 10 | 1.0000/1.0000 | 1.0000/1.0000 | 0.0000 [0.0000, 0.0000] | 0.00% [0.00, 0.00]% |
| async_aligned/200 minus async_aligned/0 | all | environment steps | 30 | 136.6000/134.5000 | 142.7333/130.5000 | 6.1333 [-6.4000, 28.1675] | 4.71% [-4.61, 21.21]% |
| async_aligned/200 minus async_aligned/0 | all | wall-clock episode seconds | 30 | 9.2349/9.0702 | 10.0753/8.8973 | 0.8404 [-0.1739, 2.2462] | 8.58% [-1.68, 23.47]% |
| async_aligned/200 minus async_aligned/0 | all | episode p50 observation-to-delivery seconds | 30 | 0.1309/0.1307 | 0.4569/0.4578 | 0.3260 [0.3233, 0.3286] | 249.66% [244.56, 254.67]% |
| async_aligned/200 minus async_aligned/0 | all | episode mean adjacent-action 7D L2 | 30 | 0.1584/0.1428 | 0.1827/0.1506 | 0.0243 [-0.0124, 0.0601] | 39.46% [7.53, 77.53]% |
| async_aligned/200 minus async_aligned/0 | all | binary episode success | 30 | 1.0000/1.0000 | 1.0000/1.0000 | 0.0000 [0.0000, 0.0000] | 0.00% [0.00, 0.00]% |
| async_aligned/200 minus async_aligned/0 | task 0 | environment steps | 10 | 151.4000/151.5000 | 144.9000/147.0000 | -6.5000 [-9.3000, -3.6000] | -4.25% [-6.12, -2.35]% |
| async_aligned/200 minus async_aligned/0 | task 0 | wall-clock episode seconds | 10 | 10.2811/9.9139 | 11.0646/9.7623 | 0.7836 [-0.6098, 3.1821] | 5.93% [-5.71, 25.60]% |
| async_aligned/200 minus async_aligned/0 | task 0 | episode p50 observation-to-delivery seconds | 10 | 0.1295/0.1283 | 0.4576/0.4578 | 0.3281 [0.3264, 0.3298] | 253.56% [248.77, 257.91]% |
| async_aligned/200 minus async_aligned/0 | task 0 | episode mean adjacent-action 7D L2 | 10 | 0.1501/0.1584 | 0.1626/0.1426 | 0.0124 [-0.0622, 0.0899] | 51.09% [-20.40, 140.17]% |
| async_aligned/200 minus async_aligned/0 | task 0 | binary episode success | 10 | 1.0000/1.0000 | 1.0000/1.0000 | 0.0000 [0.0000, 0.0000] | 0.00% [0.00, 0.00]% |
| async_aligned/200 minus async_aligned/0 | task 1 | environment steps | 10 | 133.6000/131.5000 | 164.6000/130.5000 | 31.0000 [-4.9000, 90.1075] | 23.12% [-3.71, 67.91]% |
| async_aligned/200 minus async_aligned/0 | task 1 | wall-clock episode seconds | 10 | 8.8266/8.7748 | 10.7146/8.8827 | 1.8880 [-0.0970, 4.9910] | 21.45% [-1.03, 57.10]% |
| async_aligned/200 minus async_aligned/0 | task 1 | episode p50 observation-to-delivery seconds | 10 | 0.1307/0.1293 | 0.4565/0.4597 | 0.3258 [0.3206, 0.3309] | 249.89% [240.62, 257.97]% |
| async_aligned/200 minus async_aligned/0 | task 1 | episode mean adjacent-action 7D L2 | 10 | 0.1446/0.1443 | 0.1928/0.1665 | 0.0482 [-0.0049, 0.0993] | 54.30% [3.49, 121.39]% |
| async_aligned/200 minus async_aligned/0 | task 1 | binary episode success | 10 | 1.0000/1.0000 | 1.0000/1.0000 | 0.0000 [0.0000, 0.0000] | 0.00% [0.00, 0.00]% |
| async_aligned/200 minus async_aligned/0 | task 2 | environment steps | 10 | 124.8000/123.5000 | 118.7000/118.5000 | -6.1000 [-9.8000, -3.1000] | -4.75% [-7.38, -2.51]% |
| async_aligned/200 minus async_aligned/0 | task 2 | wall-clock episode seconds | 10 | 8.5969/8.5137 | 8.4465/8.4382 | -0.1503 [-0.3680, 0.0570] | -1.62% [-4.02, 0.73]% |
| async_aligned/200 minus async_aligned/0 | task 2 | episode p50 observation-to-delivery seconds | 10 | 0.1324/0.1336 | 0.4565/0.4564 | 0.3241 [0.3188, 0.3297] | 245.53% [235.29, 257.17]% |
| async_aligned/200 minus async_aligned/0 | task 2 | episode mean adjacent-action 7D L2 | 10 | 0.1805/0.1354 | 0.1927/0.1416 | 0.0122 [-0.0422, 0.0565] | 12.97% [-7.97, 35.34]% |
| async_aligned/200 minus async_aligned/0 | task 2 | binary episode success | 10 | 1.0000/1.0000 | 1.0000/1.0000 | 0.0000 [0.0000, 0.0000] | 0.00% [0.00, 0.00]% |

## Historical replacement-boundary audit

180 traces and 2477 inference events contain 2122 accepted replacements. Explicit replacement-index fields: `[]`.

Historical M3 events have observation_control_step and merge age but no explicit accepted-chunk replacement index, so boundary component jumps were not reconstructed.

## Interpretation boundary

All 180 episodes succeeded. Success-rate differences are ceilinged and do not establish equal robustness.
