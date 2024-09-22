import matplotlib.pyplot as plt
import pacmap
import numpy as np  # Для работы с массивами
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from mammoth_data import data  # Импортируем список из файла mammoth_data.py
from quick_test import test_data
from tqdm import tqdm  # Импортируем библиотеку для прогресс-бара
from umap import UMAP


# Я пытался скачать trimap (pip install trimap), но она требует Build Tools для С++, но сколько бы не устанавливал - всё равно не работает.
# Вроде этот код эмулирует алгоритм (нашел в интернете)
def manual_trimap(data, n_neighbors=5, n_components=2, n_iter=500):
    # Получение ближайших соседей
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(data)
    distances, indices = nbrs.kneighbors(data)

    # Инициализация 2D координат случайными значениями
    embedding = np.random.rand(data.shape[0], n_components)

    # Основной цикл оптимизации
    for _ in range(n_iter):
        for i in range(data.shape[0]):
            # Соседи текущей точки
            neighbors = indices[i][1:]  # исключаем саму точку

            for j in neighbors:
                # Обновление координат на основе расстояний
                diff = embedding[i] - embedding[j]
                dist = np.linalg.norm(diff)

                if dist > 0:
                    # Корректируем координаты
                    embedding[i] -= (diff / dist) * (dist - distances[i][1])  # используем distances[i][1]

    return embedding


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
        data_array = np.array(test_data)

        print("Применение TriMap (реализованного вручную) для снижения размерности до 2D")
        data_2d = manual_trimap(data_array)

        print("Визуализация результатов")
        plt.figure(figsize=(10, 8))

        print("Отрисовка всех точек")
        plt.scatter(data_2d[:, 0], data_2d[:, 1], s=10, c=data_2d[:, 0], cmap='viridis')

        plt.title('2D Visualization of Data using Manual TriMap')
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
