import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
from tensorflow.keras.utils import to_categorical

def load_model_and_tokenizer():
    # 使用完整的文件路径
    model_path = r'D:\毕业论文\抗菌肽\抗菌肽模型\独热5000个被证实.keras'
    tokenizer_path = r'D:\毕业论文\抗菌肽\抗菌肽模型\独热tokenizer.pickle'

    # 加载模型和 tokenizer
    model = load_model(model_path)
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)

    return model, tokenizer

def predict_file(input_path, output_path, model, tokenizer):
    df = pd.read_excel(input_path)
    sequences = df['Sequence'].astype(str).values
    sequences_encoded = tokenizer.texts_to_sequences(sequences)
    sequences_padded = pad_sequences(sequences_encoded, maxlen=100)
    sequences_one_hot = to_categorical(sequences_padded, num_classes=len(tokenizer.word_index) + 1)
    predictions = model.predict(sequences_one_hot)
    df['Prediction'] = predictions
    df.to_excel(output_path, index=False)
    messagebox.showinfo("Success", "预测结果已成功保存至 " + output_path)

def open_file(model, tokenizer):
    input_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
    output_path = filedialog.asksaveasfilename(filetypes=[("Excel files", "*.xlsx *.xls")], defaultextension=".xlsx")
    if input_path and output_path:
        predict_file(input_path, output_path, model, tokenizer)

app = tk.Tk()
app.title("Excel Predictor")
app.geometry("300x150")

model, tokenizer = load_model_and_tokenizer()

button = tk.Button(app, text="Open Excel File", command=lambda: open_file(model, tokenizer))
button.pack(expand=True)

app.mainloop()
