# Self-Composing Policies for Scalable Continual Reinforcement Learning

This repository is part of the supplementary material of the paper [*Self-Composing Policies for Scalable Continual Reinforcement Learning*](https://openreview.net/pdf?id=f5gtX2VWSB). The paper is published at [ICML 2024](https://icml.cc/virtual/2024/poster/33472) and selected for [oral presentation](https://icml.cc/virtual/2024/oral/35492).

To cite this project in publications:

```bibtex
@inproceedings{malagon2024selfcomp,
  title={Self-Composing Policies for Scalable Continual Reinforcement Learning},
  author={Malagon, Mikel and Ceberio, Josu and Lozano, Jose A},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2024}
}
```

## Structure of the repo 🌳

The repository is organized into three main parts: `componet`, that
holds the implementation of the proposal of the paper;
`experiments/atari`, where the experiments of the SpaceInvaders and
Freeway sequences are located; and `experiments/meta-world`, that
contains the experiments of the Meta-World sequence.

<details>

<summary>Click here to unfold the structure 🌳 of the repo.</summary>

```bash
├── componet/ # The implementation of the proposed CompoNet architecture
│
├── experiments/
│   ├── atari/        # Contains all the code related to the SpaceInvaders and Freeway sequences
│   │   ├── data.tar.xz # Contains the compressed CSV files used for the figures
│   │   ├── models/   # Implements PPO agents for all of the considered methods
│   │   ├── process_results.py  # Processes the runs generating the metrics and plots
│   │   ├── run_experiments.py  # Utility script to call `run_ppo.py` for multiple settings
│   │   ├── run_ppo.py          # Main script to run the PPO experiments
│   │   ├── task_utils.py       # Implements several task-related utils
│   │   ├── test_agent.py       # Main script to evaluate trained agents
│   │   ├── plot_ablation_input_head.py  # Plots input attention head ablation results
│   │   ├── plot_ablation_output_head.py # Plots output attention head ablation results
│   │   ├── plot_arch_val.py      # Plots architecture validation results
│   │   ├── plot_dino_vs_cnn.py   # Plots results of the comparison between DINO and CNN-based agents
│   │   ├── transfer_matrix.py    # Computes and plots the transfer matrices of SpaceInvaders and Freeway
│   │   └── requirements.txt      # Requirements file for these experiments
│   │
│   └── meta-world/          # Contains all the experiments in the Meta-World tasks
│       ├── data.tar.xz      # Contains the compressed CSV files used for the figures
│       ├── benchmarking.py  # Benchmarks CompoNet and ProgNet and plots the results
│       ├── models/          # Contains the implementations of the SAC agents
│       ├── process_results.py    # Processes the runs generating the metrics and plots
│       ├── run_experiments.py    # Utility script for running experiments
│       ├── run_sac.py            # Main script to run SAC experiments
│       ├── tasks.py              # Contains the definitions of the tasks
│       ├── test_agent.py         # Main script used to test trained agents
│       ├── transferer_matrix.py  # Computes and plots the transfer matrix of Meta-World
│       └── requirements.txt      # Requirements file for these experiments
│
├── utils/    # Contains utilities used across multiple files
├── LICENSE   # Text file with the license of the repo
└── README.md
```
</details>

Note that all of the CLI options available in the training and
visualization (`plot_*`) scripts can be seen using `--help`.

Finally, all PPO and SAC scripts are based on the excellent
[CleanRL](https://github.com/vwxyzjn/cleanrl) project, that provides
high-quality implementations of many RL algorithms.

## Requirements 📋

Likewise the experimentation, the requirements are divided in two
sets, each containing the packages required for each group of
experiments: `experiments/atari/requirements.txt` and
`experiments/meta-world/requirements.txt`.

To install the requirements:

```setup
pip install -r experiments/atari/requirements.txt
```

or,

```setup
pip install -r experiments/meta-world/requirements.txt
```

Note that the `atari` experiments use the `ALE` environments from the
[gymnasium](https://gymnasium.farama.org/) project, while `meta-world`
employs [meta-world](https://github.com/Farama-Foundation/Metaworld).


## Reproducing the results 🔄

If you want to reproduce any of the results that appear in the paper,
just call the corresponding training script with the default CLI
options (just change the environment and task options if needed).

All of the CLI options have the default value that was used in the
paper ☺️.

## License 🐃

This repository is distributed under the terms of the GLPv3
license. See the [LICENSE](./LICENSE) file for more details, or visit
the [GPLv3 homepage](https://www.gnu.org/licenses/gpl-3.0.en.html).
