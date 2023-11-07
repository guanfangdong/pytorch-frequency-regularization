# Frequency Regularization: Reducing Information Redundancy in Convolutional Neural Networks

In this repository, we introduce a cutting-edge technique termed "Frequency Regularization" that remarkably compresses a UNet model from 31 million parameters down to a mere 759 non-zero parameters without compromising on performance. This technique has been meticulously tailored and evaluated, achieving an impressive Dice Score of over 97% on the [Carvana Image Masking Challenge dataset](https://www.kaggle.com/c/carvana-image-masking-challenge). The original UNet model has a size of 366MB, but with our frequency regularization applied, the size is drastically reduced to 4KB, as demonstrated in the `unet_fr.tar.xz` file included in this demo.

## Getting Started

### Prerequisites

- Ensure that you have the `imageio` package installed for loading testing images. Follow the [installation instructions](https://imageio.readthedocs.io/en/v2.8.0/installation.html) if you don't have it installed yet.

### Running the Demo

1. Clone this repository to your local machine.
2. Navigate to the project directory.
3. Execute the following command to run the demo:
   ```bash
   bash run.sh
   ```

### Additional Information

Due to the double-blind policy, comments within the code have been omitted. However, the demonstration is structured to be straightforward and the execution command provided should seamlessly run the demo.

Feel free to reach out for any inquiries or further clarifications regarding the implementation and performance of the frequency regularization technique on the UNet model.

## Current and Future Plans

| Milestones                  | Status        |
|-----------------------------|---------------|
| Initial U-net Model        | ✔️ Completed   |
| Additional Model Releases  | 🔜 Upcoming   |
| Example Training Code      | 🔜 Upcoming   |
| Pip Repository Creation    | ✔️ Completed   |


## Current Progress
The initial model utilizing the U-net architecture has been developed and rigorously validated. This model serves as a compelling testament to the power of Frequency Regularization to markedly diminish model size without sacrificing performance integrity. 

## Upcoming Releases
Future efforts will focus on disseminating a variety of models with accompanying training code to elucidate the breadth and potential of Frequency Regularization. These models will span multiple architectures and datasets, affirming the method's generalizability and strength.

## Pip Repository
We have accomplished the development of a pip repository for our Frequency Regularization technique. You can now integrate Frequency Regularization into your projects by simply running `pip install frereg`. This step is instrumental in simplifying the deployment of condensed yet potent models in pragmatic applications.

# Installation and Running the Demo With the Pip Repository

With the newly created pip repository, installation and execution have become more streamlined. Please ensure you have completed the following steps to use the Frequency Regularization technique with your UNet models:

### Installation via Pip Repository

You can install the Frequency Regularization package by executing the following command in your environment:

```bash
pip install frereg
```

This command will install the necessary components to apply Frequency Regularization to your convolutional neural network models.


# Citation

If you find our Frequency Regularization technique intriguing or utilize it in your research, we kindly encourage you to cite our paper:

```bibtex
@ARTICLE{10266314,
  author={Zhao, Chenqiu and Dong, Guanfang and Zhang, Shupei and Tan, Zijie and Basu, Anup},
  journal={IEEE Access},
  title={Frequency Regularization: Reducing Information Redundancy in Convolutional Neural Networks},
  year={2023},
  volume={11},
  number={},
  pages={106793-106802},
  doi={10.1109/ACCESS.2023.3320642}
}
```

The paper is published in IEEE Access, and you can access it [here](https://ieeexplore.ieee.org/abstract/document/10266314).

Title: [Frequency Regularization: Reducing Information Redundancy in Convolutional Neural Networks](https://ieeexplore.ieee.org/abstract/document/10266314)

Your acknowledgment greatly supports our ongoing research and contributes to fostering advancements in network compression techniques.
