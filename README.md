# Plume
**Plume is the training framework behind Gelato, the state-of-the-art adaptive bitrate (ABR) controller on [Puffer](https://puffer.stanford.edu) ([GitHub](https://github.com/StanfordSNR/puffer)).** Puffer is Stanford's free, open-source live TV streaming platform and a real-world research testbed for improving video streaming, evaluating ABR controllers in the real world by streaming to 350,000+ users across the wide area Internet.

Trained with Plume and evaluated on Puffer for more than a year, Gelato still outperforms all evaluated ML controllers, nearly 4 years afterwards. It was the first controller on the platform to deliver statistically significant improvements in both video quality and stalling, reducing stalls by as much as 75%. The work is described in [*Practically High-Performant Neural Adaptive Video Streaming*](https://dl.acm.org/doi/10.1145/3696401), winner of the **Best Paper Award at ACM CoNEXT 2024**.

## Plume and Gelato

Network-trace datasets are highly skewed: common network conditions dominate training, while difficult but important conditions are underrepresented. Plume is a general training framework that addresses this problem in three stages:

1. Identify the trace features that most strongly affect a controller's behavior.
2. Cluster traces using those features.
3. Prioritize salient clusters during training so the controller learns from a more useful balance of network conditions.

Gelato is the neural ABR controller trained with this framework. It uses playback and network observations to select the quality of each upcoming video chunk, balancing video quality against the risk of rebuffering. Plume's contribution is not a Puffer-specific heuristic: it is a systematic way to build better training distributions for learning-based controllers, with Gelato demonstrating the approach at scale on Puffer.

## Repository layout

This repository provides [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) environments and training code for reproducing the controlled and Puffer-based experiments from the paper. Parts of the implementation build on the [Park project](https://github.com/park-project/park/tree/master/park/envs/abr_sim), [Pensieve](https://github.com/hongzimao/pensieve), and [Puffer](https://github.com/StanfordSNR/puffer).

- [`controlled_abr`](controlled_abr) contains Trace-Bench, the controlled environment used for repeatable experiments over synthetically generated trace distributions.
- [`puffer_abr`](puffer_abr) contains the trace-driven simulation environment built from Puffer's public logs, along with the code used to train and evaluate Gelato.

Each directory has its own README with installation, data preparation, and training instructions.

## Train and test on Puffer

Use [`puffer_abr`](puffer_abr) to download or load Puffer traces, train Gelato with Plume's sampling strategies, and evaluate controllers in simulation.

For the models used by Gelato and the adapter that connects Python ABR controllers to a Puffer deployment, see [`sagar-pa/abr_rl_test`](https://github.com/sagar-pa/abr_rl_test). That repository also provides the interface for adding another controller and testing it against Puffer.

Useful Puffer links:

- [Puffer live website](https://puffer.stanford.edu)
- [Puffer source code](https://github.com/StanfordSNR/puffer)
- [Gelato models and Puffer testing adapter](https://github.com/sagar-pa/abr_rl_test)

## Citation

If you use this code, the Gelato models, or the Puffer adapter as part of a research project, please cite the original paper:

```BibTeX
@article{plume2024,
author = {Patel, Sagar and Zhang, Junyang and Narodystka, Nina and Jyothi, Sangeetha Abdu},
title = {Practically High Performant Neural Adaptive Video Streaming},
year = {2024},
issue_date = {December 2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {2},
number = {CoNEXT4},
url = {https://doi.org/10.1145/3696401},
doi = {10.1145/3696401},
journal = {Proc. ACM Netw.},
month = nov,
articleno = {30},
numpages = {23},
keywords = {deep reinforcement learning, video streaming}
}
```
