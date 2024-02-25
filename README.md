**About the Project**

Multimodal sentiment analysis involves analyzing and understanding sentiment from
multiple sources of data. In traditional sentiment analysis, the focus is mainly on text,
but in multimodal approaches, the analysis extends to images, videos, audio, and
other modalities. For this project I have chosen to work with Text and Visual Data. The
aim is to analyze how well different modalities work in combination to determine the
tone of a person.
For this analysis I have chosen the MELD dataset which is a common dataset used to
benchmark such models.
Throughout the semester, we delved into Natural Language Processing and the
revolutionary role of transformers in advancing NLP research. Given my strong
interest in computer vision, I was particularly fascinated by exploring how
transformers contribute to the analysis of visual sequences.

**The Dataset**
MELD (Multimodal EmotionLines Dataset) is a dataset designed for research in
multimodal sentiment analysis. It combines textual data with audio and visual
modalities to provide a comprehensive resource for studying emotion and sentiment
in conversations. For this project, I am only using text and visual features of the
dataset. Along with this due to the limited compute, I am using 6200 utterances from
the 9990 total utterances. The dataset have 3 labels for each utterance: Neutral,
Positive and Negative.

**Model Specifications**
Text Model : For textual sentiment analysis I am using RoBERTa with a vocab size of
50265 and hidden size of 768, with 12 hidden layers and 12 attention heads.
Visual Model : For visual sentiment analysis I am using VideoMAE, which takes
multiple frames as an input and encodes them to a vector. I am using the model with 8
frames per video (frame size : 224x224), hidden size of 768, with 12 hidden layers and
12 attention heads.
Multimodal Model : For multimodal sentiment analysis, I am using both RoBERTa
along with VideoMAE combined, along with an added Linear layer to create a
classifier. I have concatenated the logits from both text and visual models to input to
the liner layer.

**Code Overview**
--learning_rate : 'The Learning Rate of the Model'
--batch_size : 'The batch size for training'
--num_frames : 'Num of frames in the data preprocessing step for
’ visual model’
--epochs : 'Total number of epochs for training'
--iteration : 'Iteration for creating data loader'
--dropout : 'Dropout value for al the models'
--model : 'Which model to run visual/text/multimodal'
