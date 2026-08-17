# X-VLA native-Isaac input counterfactual

Status: completed offline first-chunk diagnostic; this is not an episode success result.

| Frozen comparison | Full chunk RMSE | First action XYZ L2 | First action 7D L2 |
| --- | ---: | ---: | ---: |
| Native vs official images, official state held | 0.761554 | 0.079303 | 2.001838 |
| Native vs official state, official images held | 0.005231 | 0.013777 | 0.013811 |
| Fully native vs fully official | 0.761857 | 0.079195 | 2.001831 |

All values are paired across the three frozen inference seeds. Interpret larger action-space displacement as stronger input sensitivity, not as proof of task-level causality. The exact actions and per-seed values are in `summary.json`.
