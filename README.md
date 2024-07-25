Autoencoder Trainer
Overview

* This project includes a Python implementation of an autoencoder model to reduce dimensionality of hyperspectral leaf reflectance data.
* The model is designed to preprocess the data, build an autoencoder model, and train it with a sample dataset.
* Cite:
Tross, M. C, Grzybowski, M. W, Jubery, T. Z, Grove, R. J, Nishimwe, A. V, Torres-Rodriguez, J. V., Sun, G., Ganapathysubramanian, B., Ge, Y., & Schnable, J. C. (2024). Data driven discovery and quantification of hyperspectral leaf reflectance phenotypes across a maize diversity panel. The Plant Phenome Journal, 7, e20106. https://doi.org/10.1002/ppj2.20106

Getting Started
Prerequisites

* Python 3.x
* Pip package manager

Installation

1. Clone the Repository
    * Run the following commands:

    	git clone https://github.com/mtross2/autoencoder_hyperspec_ref.git

    	cd autoencoder_hyperspec_ref

2. Set Up a Virtual Environment (Optional but recommended)

    * For Windows:

        python -m venv venv

        .\venv\Scripts\activate

    * For Unix or MacOS:
    
        python3 -m venv venv

        source venv/bin/activate

3. Install Required Packages

    * Execute the command:

        pip install -r requirements.txt

4. Install Your Package (Optional if you want to use it as a package)

    * Use this command:

        python setup.py install

Training the Model

    * Prepare Your Dataset
        * Place your dataset in the sample_data folder.
        * Ensure it is in the correct format as expected by the AutoencoderTrainer.

    * Run the Training Script
        * Run the training script with default parameters or provide custom values.
            python scripts/train_autoencoder.py --dataset sample_data/sampledData_maize2020.csv.gz

    * To see all available command-line options:
	python scripts/train_autoencoder.py --help

Testing

    * Run the test suite to ensure everything is set up correctly:
        python -m unittest
