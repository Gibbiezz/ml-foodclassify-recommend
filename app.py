import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import json

app = Flask(__name__)

# Load the pre-trained model
model = load_model('classifier_finetuned.h5')

# Load recipe data from JSON file
with open('recipes.json', 'r') as f:
    recipes = json.load(f)

# Fungsi untuk mencari resep yang relevan berdasarkan 'Nama Resep'
def find_related_recipes(ingredient):
    related_recipes = []
    ingredient = ingredient.lower()  # Mengubah label menjadi huruf kecil agar lebih fleksibel
    
    for recipe in recipes:
        # Cek jika nama bahan yang terdeteksi ada di 'Nama Resep'
        if ingredient in recipe["Nama Resep"].lower():
            related_recipes.append(recipe)
    return related_recipes

# Fungsi untuk mengklasifikasikan gambar
def classify_image(image_path):
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((224, 224))  # Sesuaikan ukuran input dengan model
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    label_index = np.argmax(prediction, axis=1)[0]

    class_names = ['Ayam', 'Bayam', 'Brokoli', 'Daging', 'Ikan', 'Kentang', 'Tahu', 'Telur', 'Tempe', 'Tomat', 'Udang', 'Wortel']
    predicted_label = class_names[label_index]

    return predicted_label

# Route utama
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'image' not in request.files:
            return "No file part"
        
        file = request.files['image']
        if file.filename == '':
            return "No selected file"

        # Validasi ekstensi file
        allowed_extensions = {'png', 'jpg', 'jpeg'}
        if not any(file.filename.endswith(ext) for ext in allowed_extensions):
            return "Invalid file format. Please upload a PNG, JPG, or JPEG file."

        # Simpan file gambar
        if not os.path.exists('uploads'):
            os.makedirs('uploads')

        image_path = os.path.join('uploads', file.filename)
        file.save(image_path)
        
        # Prediksi bahan dari gambar
        try:
            predicted_label = classify_image(image_path)
        except Exception as e:
            return f"Error in image classification: {e}"
        
        # Cari resep terkait dengan bahan yang diprediksi
        related_recipes = find_related_recipes(predicted_label)
        
        return render_template("index.html", label=predicted_label, recipes=related_recipes)

    return render_template("index.html", label=None, recipes=[])

if __name__ == "__main__":
    app.run(debug=True)
