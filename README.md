# Frequency Regularization-based UNet Model

In this repository, we introduce a cutting-edge technique termed "Frequency Regularization" that remarkably compresses a UNet model from 31 million parameters down to a mere 759 non-zero parameters without compromising on performance. This technique has been meticulously tailored and evaluated, achieving an impressive Dice Score of over 97% on the [Carvana Image Masking Challenge dataset](https://www.kaggle.com/c/carvana-image-masking-challenge). The original UNet model has a size of 366MB, but with our frequency regularization applied, the size is drastically reduced to 40KB, as demonstrated in the `unet_fr.pt` file included in this demo.

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
