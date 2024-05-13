# Differentiable-design-of-discrete-optics-with-gumbel-softmax-trick
Very simple Pytorch implementation of gumbel-softmax trick for differentiable design of discrete optics
- such as the example of **designing multi-level doe for hologram generation** shown below.

Credit to [Cheng Zheng @ MIT](https://github.com/zcshinee) debugs and makes the DOE with gumbel-softmax module work. 
## How to start:
**Just run main.py**

![Alt Text](https://media.giphy.com/media/VbnUQpnihPSIgIXuZv/giphy-downsized.gif))

## Example:
- Optimize a size of $32\times32$ multilevel ($4$) diffraction optical lements (DOE) to render the img on the right.
![alt-text](plots/optimization_evolution.gif)

**Final doe profile in 3D is:** 

![alt-text](plots/final_doe_profile.png)

## Cite it:
The module here was developed for our project and paper [Neural Lithography](https://neural-litho.github.io/).
If you find our code or any of our materials useful, please cite our work:

```bibtex
@article{zheng2023neural,
            title={Neural Lithography: Close the Design-to-Manufacturing Gap in Computational Optics with a'Real2Sim'Learned Photolithography Simulator},
            author={Zheng, Cheng and Zhao, Guangyuan and So, Peter TC},
            journal={arXiv preprint arXiv:2309.17343},
            year={2023}
            }
```


```bibtex
@inproceedings{zheng2023close,
            title={Close the Design-to-Manufacturing Gap in Computational Optics with a'Real2Sim'Learned Two-Photon Neural Lithography Simulator},
            author={Zheng, Cheng and Zhao, Guangyuan and So, Peter},
            booktitle={SIGGRAPH Asia 2023 Conference Papers},
            pages={1--9},
            year={2023}
}
```


