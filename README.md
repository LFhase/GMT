
<h1 align="center">GMT: Graph Multilinear neT</h1>
<p align="center">
    <a href="https://github.com/LFhase/GM"><img src="https://img.shields.io/badge/arXiv-xxxx.xxxx-b31b1b.svg" alt="Paper"></a>
    <a href="https://github.com/LFhase/GMT"><img src="https://img.shields.io/badge/-Github-grey?logo=github" alt="Github"></a>
    <!-- <a href="https://colab.research.google.com/drive/1t0_4BxEJ0XncyYvn_VyEQhxwNMvtSUNx?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab"></a> -->
    <a href="https://openreview.net/forum?id=A6AFK_JwrIW"> <img alt="License" src="https://img.shields.io/static/v1?label=Pub&message=ICML%2724&color=blue"> </a>
    <a href="https://github.com/LFhase/GMT/blob/main/LICENSE"> <img alt="License" src="https://img.shields.io/github/license/LFhase/CIGA?color=blue"> </a>
    <!-- <a href="https://icml.cc/virtual/2024/poster/3455"> <img src="https://img.shields.io/badge/Video-grey?logo=Kuaishou&logoColor=white" alt="Video"></a> -->
    <!-- <a href="https://lfhase.win/files/slides/GMT.pdf"> <img src="https://img.shields.io/badge/Slides-grey?&logo=MicrosoftPowerPoint&logoColor=white" alt="Slides"></a> -->
   <!--  <a href="https://icml.cc/media/PosterPDFs/ICML%202022/a8acc28734d4fe90ea24353d901ae678.png"> <img src="https://img.shields.io/badge/Poster-grey?logo=airplayvideo&logoColor=white" alt="Poster"></a> -->
</p>

This repo contains the sample code for reproducing the results of our ICML 2024 paper: *[How Interpretable Are Interpretable Graph Neural Networks?](https://openreview.net/forum?id=A6AFK_JwrIW)*, which has also been presented as ***spotlight*** at [ICLR MLGenX](https://openreview.net/group?id=ICLR.cc/2024/Workshop/MLGenX). 😆😆😆

Updates:

- [ ] Camera-ready version of the paper will be updated soon!
- [ ] Full code and instructions will be released soon!

## Regular Graphs

`/GSAT` contains the codes for running on regular graphs

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

## Geometric Graphs

`/LRI` contains the codes for running on geometric graphs

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
