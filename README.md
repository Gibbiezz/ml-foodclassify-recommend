```markdown
# **ML Food Classify & Recipe Recommendation**

Welcome to **ML Food Classify & Recipe Recommendation**! This project uses machine learning to classify food images and recommend recipes based on the predicted class.

## **How to Use This Repository**

Follow these steps to get everything set up and running:

### **1. Clone the repository**
Start by cloning this repository to your local machine:

```bash
git clone https://github.com/Gibbiezz/ml-foodclassify-recommend.git

### **2. Set up your Google Colab environment**
If you're using Google Colab, here’s what you need to do:
- Upload the dataset.zip file to your Google Drive.
- In Colab, ensure the following files are available:
- classifier_finetuned.h5 (your fine-tuned model)
- dataset.zip (the dataset file to train your model)
- imageclassify.ipynb (Jupyter notebook for classification)
- recipes.json (JSON file with food recipes)
- test images/ (folder with test images like test12.jpeg)

### **3. Extract and split the dataset**
After uploading the dataset, you’ll need to extract the dataset.zip file and split the images into training and validation sets (80%/20%). The Colab notebook already contains the code to handle this, so just follow the steps in the notebook.

### **4. Run the notebook**
Open the imageclassify.ipynb notebook and run the cells in order. The notebook will guide you through the process of training the model, predicting the class of food images, and recommending recipes based on the predicted class.

### **5. Test the model**
Once the model is ready, you can upload a test image (such as one from the test images/ folder). The notebook will classify the image and recommend a recipe based on the class predicted by the model.
