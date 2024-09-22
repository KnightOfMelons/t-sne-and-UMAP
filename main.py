import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mammoth_data import data  # Импортируем список из файла mammoth_data.py
from quick_test import test_data
import numpy as np  # Для работы с массивами
from tqdm import tqdm  # Импортируем библиотеку для прогресс-бара
from umap import UMAP

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
        pass
    elif choose == 4:
        pass
    elif choose == 0:
        break
    else:
        continue
