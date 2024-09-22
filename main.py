import matplotlib.pyplot as plt
import pacmap
import numpy as np  # Для работы с массивами
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from mammoth_data import data  # Импортируем список из файла mammoth_data.py
from quick_test import test_data
from tqdm import tqdm  # Импортируем библиотеку для прогресс-бара
from umap import UMAP
from trimap import TRIMAP


while True:
    choose = int(input("\n1 - Алгоритм t-sne.\n2 - Алгоритм UMAP.\n3 - Алгоритм TriMap.\n4 - Алгоритм PaCMAP.\n0 - Выход.\n\nВаш выбор: "))

    if choose == 1:
        print("Преобразуем данные в массив NumPy")
        # Я везде поставил пока test_data из quick_test.py, там находится меньше значений (где-то 40 тысяч), посему визуализирование функционала
        # происходит быстрей, можете сменить на data, чтобы тестировать из полного списка mammoth_data.py.
        data_array = np.array(test_data)

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

    elif choose == 2:
        print("Преобразуем данные в массив NumPy")
        # Я везде поставил пока test_data из quick_test.py, там находится меньше значений (где-то 40 тысяч), посему визуализирование функционала
        # происходит быстрей, можете сменить на data, чтобы тестировать из полного списка mammoth_data.py.
        data_array = np.array(test_data)

        print("Применение UMAP для снижения размерности до 2D")
        umap = UMAP(n_components=2, random_state=42, verbose=True)

        print("Применяем UMAP")
        data_2d = umap.fit_transform(data_array)

        print("Визуализация результатов")
        plt.figure(figsize=(10, 8))

        print("Отрисовка всех точек")
        plt.scatter(data_2d[:, 0], data_2d[:, 1], s=10, c=data_2d[:, 0], cmap='viridis')

        plt.title('2D Visualization of Data using UMAP')
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        plt.grid(True)
        plt.show()

    elif choose == 3:
        print("Преобразуем данные в массив NumPy")
        data_array = np.array(data)

        # Инициализируем модель TriMap без параметров
        trimap_model = TRIMAP()

        print("Применяем TriMap")
        data_2d = trimap_model.fit_transform(data_array)  # Попробуем без дополнительных параметров

        print("Визуализация результатов")
        plt.figure(figsize=(10, 8))

        print("Отрисовка всех точек")
        plt.scatter(data_2d[:, 0], data_2d[:, 1], s=10, c=data_2d[:, 0], cmap='viridis')

        plt.title('2D Visualization of Data using TriMap')
        plt.xlabel('TriMap Component 1')
        plt.ylabel('TriMap Component 2')
        plt.grid(True)
        plt.show()
                
    elif choose == 4:
        print("Преобразуем данные в массив NumPy")
        data_array = np.array(test_data)

        print("Применение PaCMAP для снижения размерности до 2D")
        pacmap_model = pacmap.PaCMAP(n_components=2, random_state=42, verbose=True)

        print("Применяем PaCMAP")
        data_2d = pacmap_model.fit_transform(data_array)

        print("Визуализация результатов")
        plt.figure(figsize=(10, 8))

        print("Отрисовка всех точек")
        plt.scatter(data_2d[:, 0], data_2d[:, 1], s=10, c=data_2d[:, 0], cmap='viridis')

        plt.title('2D Visualization of Data using PaCMAP')
        plt.xlabel('PaCMAP Component 1')
        plt.ylabel('PaCMAP Component 2')
        plt.grid(True)
        plt.show()

    elif choose == 0:
        break
    else:
        continue
