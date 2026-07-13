# EAST-SPL: Event-Aware Statistical Tiling for Decomposable Soccer Player Localization with an auxiliary rejection network
**[Abolfazl Chaman Motlagh](https://github.com/AbolfazlChM95)**, **[Mikael Nilsson](https://portal.research.lu.se/sv/persons/mikael-nilsson-2/)**

Official Repository of "EAST-SPL: Event-Aware Statistical Tiling for Decomposable Soccer Player Localization with an auxiliary rejection network". Accepted to International Conference on Pattern Recognition (ICPR) 2026. Lyon, France


## Overview

EAST-SPL replaces static single-frame tiling with an event-aware objective that minimizes **Expected Total FLOPs**. It uses player-location statistics to allocate finer tiles where players are more likely to appear, an **auxiliary rejection network** to skip empty tiles, and a **genetic algorithm** to optimize the tiling configuration.







## Citation
If you use this repository, please cite the main paper:
```bibtex
@inproceedings{chamanmotlagh2026eastspl,
  author    = {Chaman Motlagh, Abolfazl and Nilsson, Mikael},
  title     = {{EAST-SPL}: Event-Aware Statistical Tiling for Decomposable Soccer Player Localization with an auxiliary rejection network},
  booktitle = {Proceedings of the International Conference on Pattern Recognition},
  year      = {2026},
  note      = {Accepted}
}
```

Companion Workshop paper on Reproducibility of EAST-SPL
```bibtex
@misc{chamanmotlagh2026reproducibility,
  author = {Chaman Motlagh, Abolfazl and Nilsson, Mikael},
  title  = {On the Reproducibility of the {EAST-SPL} Pipeline: Event-Aware Statistical Tiling for Soccer Player Localization},
  year   = {2026},
  note   = {Submitted to the Sixth Workshop on Reproducible Research in Pattern Recognition}
}
```

### Related Works
This work extends our previous work [DTSP-BEV: Decomposable Tiled Soccer Player Localization↗](https://doi.org/10.5220/0014468500004067), presented at ICPRAM 2026.

Older work from our group, *"SPL-BEV: Soccer Player Localization and Birds-Eye-View Estimation"*: [[paper]](https://doi.org/10.1007/978-3-032-04968-1_10) | [[Code]](https://github.com/IvarPersson/SPL-BEV). 

### Dataset
Dataset used in this work is [Spiodeo SoccerNet SynLoc](https://github.com/Spiideo/sskit).
To cite the dataset use:
```bibtex
@inproceedings{ardo2025,
  author={Håkan Ardö and Mikael Nilsson and Anthony Cioppa and Floriane Magera and Silvio Giancola and Haochen Liu and Bernard Ghanem and Marc Van Droogenbroeck},
  booktitle={In Proceedings of the 20th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications - Volume 2: VISAPP},
  title={Spiideo SoccerNet SynLoc - Single Frame World Coordinate Athlete Detection and Localization with Synthetic Data},
  year={2025},
  pages={278-285},
  publisher={SciTePress},
  organization={INSTICC},
  issn={2184-4321},
  doi={10.5220/0013108200003912},
  isbn={978-989-758-728-3},
}
```

## License
EAST-SPL code is released under the MIT License. See [LICENSE](LICENSE) for additional details.
