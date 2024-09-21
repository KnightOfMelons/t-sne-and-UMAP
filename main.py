import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mammoth_data import data  # Импортируем список из файла mammoth_data.py
import numpy as np  # Для работы с массивами
from tqdm import tqdm  # Импортируем библиотеку для прогресс-бара

print("Преобразуем данные в массив NumPy")
data_array = np.array(data)

print("Применение t-SNE для снижения размерности до 2D с прогресс-баром")
tsne = TSNE(n_components=2, random_state=42, verbose=1)

print("Применяем t-SNE")
data_2d = tsne.fit_transform(data_array)

print("Визуализация результатов")
plt.figure(figsize=(10, 8))

print("Отрисовка всех точек")
plt.scatter(data_2d[:, 0], data_2d[:, 1], s=10, c=data_2d[:, 0], cmap='viridis')

plt.title('2D Visualization of Data using t-SNE')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(True)
plt.show()
