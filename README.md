# Real or Fake: The Imposter Hunt in Texts

## Kaggle Competition Description
"The main task of this competition is to detect fake texts in a given dataset. Each data sample contains two texts - one real and one fake. ... Importantly, both texts in each sample - real (optimal for the recipient, as close as possible to the hidden original text) and fake (more or much more distant from the hidden original text) - have been significantly modified using LLMs." Find out more information about the competition [here](https://www.kaggle.com/competitions/fake-or-real-the-impostor-hunt/overview).

## Dataset
The data used within this project is hosted in Kaggle and can be found on the [competition data page](https://www.kaggle.com/competitions/fake-or-real-the-impostor-hunt/data).

## Siamese Network Architecture - V1.6
The algorithm used is a Siamese Network that is fine tuned from the roBERTa-base model. 

## Running the Siamese Network 
Before running the model you'll first need to setup your environment with the following commands: 
```bash
python -m venv venv 
source venv/bin/activate
pip install -r requirements.txt
```

To train the Model perform the following steps: 
```bash
python dataloader.py 
python siamesetrain.py 
```

To evaluate the Model perform execute the following command: 
```bash
python siamesetest.py 
```