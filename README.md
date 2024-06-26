<h1 align="center">GMT: Graph Multilinear neT</h1>
<p align="center">
    <a href="https://github.com/LFhase/GM"><img src="https://img.shields.io/badge/arXiv-2406.07955-b31b1b.svg" alt="Paper"></a>
    <a href="https://github.com/LFhase/GMT"><img src="https://img.shields.io/badge/-Github-grey?logo=github" alt="Github"></a>
    <!-- <a href="https://colab.research.google.com/drive/1t0_4BxEJ0XncyYvn_VyEQhxwNMvtSUNx?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab"></a> -->
    <a href="https://arxiv.org/abs/2406.07955"> <img alt="License" src="https://img.shields.io/static/v1?label=Pub&message=ICML%2724&color=blue"> </a>
    <a href="https://github.com/LFhase/GMT/blob/main/LICENSE"> <img alt="License" src="https://img.shields.io/github/license/LFhase/CIGA?color=blue"> </a>
    <!-- <a href="https://icml.cc/virtual/2024/poster/3455"> <img src="https://img.shields.io/badge/Video-grey?logo=Kuaishou&logoColor=white" alt="Video"></a> -->
    <!-- <a href="https://lfhase.win/files/slides/GMT.pdf"> <img src="https://img.shields.io/badge/Slides-grey?&logo=MicrosoftPowerPoint&logoColor=white" alt="Slides"></a> -->
   <!--  <a href="https://icml.cc/media/PosterPDFs/ICML%202022/a8acc28734d4fe90ea24353d901ae678.png"> <img src="https://img.shields.io/badge/Poster-grey?logo=airplayvideo&logoColor=white" alt="Poster"></a> -->
</p>

This repo contains the sample code for reproducing the results of our ICML 2024 paper: *[How Interpretable Are Interpretable Graph Neural Networks?](https://arxiv.org/abs/2406.07955)*, which has also been presented as ***spotlight*** at [ICLR MLGenX](https://openreview.net/group?id=ICLR.cc/2024/Workshop/MLGenX). 😆😆😆

Updates:

- [X] Camera-ready version of the paper have been updated!
- [X] Full code and instructions have been released!

## Preparation

### Environment Setup

We mainly use the following key libraries with the cuda version of 11.3:

```
torch==1.10.1+cu113
torch_cluster==1.6.0
torch_scatter==2.0.9
torch_sparse==0.6.12
torch_geometric==2.0.4
```

To setup the environment, one may use the following commands under the conda environments:

```
# Create your own conda environment, then...
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
# Pytorch geometric
pip install torch_scatter==2.0.9 torch_sparse==0.6.12 torch_cluster==1.6.0 -f https://data.pyg.org/whl/torch-1.10.1+cu113.html
pip install torch_geometric==2.0.4
# Additional libraries
pip install -r requirements.txt
```

### Datasets

To prepare the datasets of regular graphs, following the [instructions in GSAT](https://github.com/Graph-COM/GSAT?tab=readme-ov-file#instructions-on-acquiring-datasets).

To prepare the datasets of geometric graphs, following the [instructions in LRI](https://github.com/Graph-COM/LRI?tab=readme-ov-file#datasets).

## Experiments on Regular Graphs

`/GSAT` contains the codes for running on regular graphs. The instructions to reproduce our results are given in [scripts/gsat.sh](scripts/gsat.sh).

### Sample Commands

For `GSAT`

```
python run_gmt.py --dataset spmotif_0.5 --backbone GIN --cuda 0 -fs 1 -mt 0
```

For `GMT-lin`

```
python run_gmt.py --dataset spmotif_0.5 --backbone GIN --cuda 0 -fs 1 -mt 3 -ie 0.5
```

For `GMT-sam`

```
# train subgraph extractor
python run_gmt.py --dataset spmotif_0.5 --backbone GIN --cuda 0 -fs 1 -mt 5 -st 200 -ie 0.5 -sm 
# train subgraph classifier
python run_gmt.py --dataset spmotif_0.5 --backbone GIN --cuda 0 -fs 1 -mt 5550 -st 200 -ie 0.5 -fm -sr 0.8
```

## Experiments on Geometric Graphs

`/LRI` contains the codes for running on geometric graphs. The instructions to reproduce our results are given in [scripts/lri.sh](scripts/lri.sh).

### Sample Commands

For `LRI-Bern`

```
python trainer.py -ba --cuda 0 --backbone egnn --dataset actstrack_2T --method lri_bern -mt 0
```

For `GMT-lin`

```
python trainer.py -ba --cuda 0 --backbone egnn --dataset actstrack_2T --method lri_bern -mt 0 -ie 0.1
```

For `GMT-sam`

```
# train subgraph extractor
python trainer.py -ba -smt 55 -ie 0.1 -fr 0.7 --cuda 0 --backbone egnn --dataset actstrack_2T --method lri_bern -mt 55 -ir 1
# train subgraph classifier
python trainer.py -ba -smt 55 -ie 0.1 -fr 0.7 --cuda 0 --backbone egnn --dataset actstrack_2T --method lri_bern -mt 5553
```

## Misc

If you find our paper and repo useful, please cite our paper:

```bibtex
@inproceedings{chen2024gmt,
    title={How Interpretable Are Interpretable Graph Neural Networks?},
    author={Yongqiang Chen and Yatao Bian and Bo Han and James Cheng},
    booktitle={International Conference on Machine Learning},
    year={2024},
    url={https://openreview.net/forum?id=F3G2udCF3Q}
}
```

We would like to acknowledge the contribution of GSAT and LRI from [Graph-COM](https://github.com/Graph-COM/) to the base codes.
