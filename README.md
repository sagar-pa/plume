# Plume
This is the public repository of the code part of the paper [*Practically High-Performant Neural Adaptive Video Streaming*](https://dl.acm.org/doi/10.1145/3696401), the **Best Paper award** winner at ACM CoNext 2024. 

This is a clean implementation of the adaptive bitrate reinforcement learning environment in [Open AI's Gym](https://github.com/openai/gym). The code is partly based off of the code in [Park Project](https://github.com/park-project/park/tree/master/park/envs/abr_sim), [Pensieve](https://github.com/hongzimao/pensieve) and [Puffer](https://github.com/StanfordSNR/puffer). 

If you are using any of this code (or any of the [deployment code](https://github.com/sagar-pa/abr_rl_test)) as part of a research project, we ask that you please cite the original paper:
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

The code is split into two directories: `controlled_abr` and `puffer_abr`. `controlled_abr` corresponds to the controlled Trace-Bench environment in the paper (where the traces are generated in a controlled manner), while `puffer_abr` is the simulation environment that uses the logs produced by [Puffer](https://puffer.stanford.edu), a free and open-source live TV streaming website and a research study at Stanford University. This is the environment that implements Gelato. 

For more details and usage, please clone the repo and see the `README.md` of those directories.

## To Do 

We will add more documentation and add more functionality in the future.
- [ ] Add function to integrate given throughput traces
- [ ] Add documentation for evaluating classical policies
- [ ] Add documentation for plotting functions given