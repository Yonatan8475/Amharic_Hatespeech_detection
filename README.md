# Text Classification with Hugging Face Transformers for Hate Speech Detection

This project focuses on training and fine-tuning a text classification model using **Hugging Face Transformers** to detect hate speech in social media posts. The model is trained to classify text into different categories such as "Hate", "Normal", and "Offensive". The objective of this project is to address the challenge of identifying harmful content in digital spaces, particularly in languages like Amharic, which is widely spoken in Ethiopia.

## Project Overview

This repository contains the following key functionalities:
1. **Data Preprocessing**: Tokenizing and balancing text data for training.
2. **Model Training**: Fine-tuning a pre-trained language model for hate speech detection.
3. **Model Evaluation**: Plotting training and validation losses, as well as visualizing the confusion matrix.
4. **Interactive Inference**: A loop that allows users to input text and get real-time predictions with confidence scores.

## Requirements

To run the code, you’ll need to have the following installed:
- Python 3.x
- Hugging Face `transformers` library
- `torch` (PyTorch)
- `scikit-learn`
- `matplotlib`
- `pandas`

You can install the dependencies using:

```bash
pip install transformers torch scikit-learn matplotlib pandas

Usage
1. Data Preprocessing
The function balance_data is used to balance the dataset by downsampling the classes. You can visualize the distribution of labels using the plot_dataset_distribution function.

2. Tokenization and Dataset Splitting
The function tokenize_and_split tokenizes the text data and splits it into training, validation, and test datasets using Hugging Face's Dataset object.


3. Model Training
To train the model, use the train_model function. It fine-tunes a pre-trained model (xlm-roberta-base in this case) for hate speech detection.


4. Evaluation
After training, you can plot the training and validation loss curves using the plot_loss function. Additionally, you can evaluate the model’s performance on the test dataset with a confusion matrix using plot_confusion_matrix.


5. Interactive Inference
The function interactive_inference allows you to interact with the trained model by typing in text and receiving predictions in real-time.

6. Displaying Predictions vs Ground Truth
You can use the display_predictions_vs_ground_truth function to compare model predictions with ground truth labels.

Hate Speech in Amharic Language
In Ethiopia, the Amharic language is widely spoken, and social media platforms have become a space for people to express themselves. However, this space is also prone to hate speech, especially in politically sensitive contexts or when tensions arise between different groups. The challenge with detecting hate speech in Amharic is the lack of sufficient labeled data and the complexity of the language itself, which may include unique expressions, insults, and culturally specific references.

Amharic hate speech can be particularly harmful when it targets individuals based on their ethnic group, political views, or social status. As social media usage grows in Ethiopia, addressing hate speech in Amharic has become crucial to creating a safer online environment.

Preventing Hate Speech
Preventing hate speech requires a multifaceted approach:

AI-Powered Detection: Building robust machine learning models that can detect harmful content automatically in multiple languages, including Amharic. This project aims to contribute by detecting hate speech in text.
User Education: Promoting awareness of what constitutes hate speech and its harmful effects on individuals and society.
Reporting Systems: Implementing more effective reporting systems on social media platforms to allow users to flag harmful content.
Content Moderation: Platforms can implement automatic content moderation that uses AI models to remove or hide harmful content.
Community Guidelines: Enforcing clear community guidelines that prohibit hate speech and encouraging users to adhere to them.
By detecting and moderating hate speech effectively, we can contribute to safer and more respectful online interactions.

Challenges and Opportunities for Improvement
Dataset Limitations
While this project has made significant strides in building a model to detect hate speech, there is a limitation in the size and diversity of the dataset used for training. The model's accuracy could be improved further if the dataset were larger and more representative of various social contexts. In particular, more data that includes diverse examples of Amharic text would enable the model to better understand the nuances of hate speech in the Ethiopian context.

The Importance of Larger Datasets
If more labeled data were available, the model could achieve better accuracy and generalize more effectively to real-world social media content. A larger dataset would allow the model to learn from more examples, including edge cases and varying expressions of hate speech, thus improving its robustness and reducing biases. Collecting more data, particularly from social media sources, would provide a richer representation of the kinds of hate speech present in different contexts.

