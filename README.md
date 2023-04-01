# Natural Language Inference on SNLI Dataset
Many thanks to my team mate [sarrabenyahia](www.github.com/sarrabenyahia) for her invaluable work.

This repository contains our work on Natural Language Inference (NLI) using the Stanford Natural Language Inference (SNLI) Dataset. Our model demonstrates an impressive AUC of 89.3%, which is significantly higher than the hazard level of 33.3%. Although the state of the art currently stands at 93%, our model showcases the potential to bridge the gap and contribute valuable insights to the NLI research community.

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Requirements](#requirements)
  - [Installation](#installation)
- [Usage](#usage)
  - [Training and Evaluating](#launching)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Our approach to NLI tackles various linguistic challenges and reduces the impact of common pitfalls associated with current techniques. The repository contains all the code, data, and resources needed to replicate our experiments and analyze our results.

## Getting Started

### Requirements

- Python 3.8 or higher
- PyTorch 1.9.0 or higher
- Transformers 4.9.2 or higher
- tqdm 4.62.0 or higher

### Installation

1. Clone the repository:
```
git clone https://github.com/Pse1234/NLI.git
```

2. Install the required packages:

```
pip install -r requirements.txt
```

## Usage

### Launching

To train the model, use the following command:
```
python model.py
```
## Results

Our model achieves an AUC of 89.3% on the SNLI Dataset. For a detailed analysis of our methodology, experiments, and results, please refer to our [paper](https://github.com/Pse1234/NLI/blob/main/Natural_Language_Inference.pdf).

## Contributing

We welcome contributions from the research community to help improve our NLI model. Please feel free to open issues, submit pull requests, or reach out to us directly.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

