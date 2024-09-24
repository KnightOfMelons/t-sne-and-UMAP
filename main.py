import matplotlib.pyplot as plt
import pacmap
import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer  # Для обработки пропусков в данных
from sklearn.preprocessing import LabelEncoder  # Добавлен импорт для LabelEncoder
from mammoth_data import data # Тут все данные
from quick_test import test_data # Тут ограниченное количество данных для быстрого теста
from tqdm import tqdm  # Импортируем библиотеку для прогресс-бара
from umap import UMAP
from ucimlrepo import fetch_ucirepo
from trimap import TRIMAP

# Все четыре алгоритма (t-SNE, UMAP, TriMap, PaCMAP) помогают нам визуализировать сложные многомерные данные в виде 2D
# или 3D картинок, чтобы было проще понять, как данные связаны между собой (или для уменьшения размерности данных как говорится в методичке).

# Обработка данных с ЛОШАДИНЫМИ КОЛИКАМИ для дальнейшей работы
def load_horse_colic_data():
    horse_colic = fetch_ucirepo(id=47)
    X = horse_colic.data.features
    y = horse_colic.data.targets
    
    return X, y, horse_colic.metadata


# Опять обрабатываю данные и очищаю там от всяких NaN, которые встречаются внутри значений
def changed_format_of_values(X, y, metadata):
        data_array = np.array(X)
        # Обработка NaN значений
        imputer = SimpleImputer(strategy='mean')
        data_array = imputer.fit_transform(data_array)

        # Кодирование целевых переменных
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y.values.flatten())  # Преобразуем в одномерный массив

        return data_array, y_encoded

X, y, metadata = load_horse_colic_data() 
data_array_horse, y_encoded_horse = changed_format_of_values(X, y, metadata)

# Основной цикл работы программы
while True:
    choose = int(input("\n1 - Алгоритм t-sne.\n2 - Алгоритм UMAP.\n3 - Алгоритм TriMap.\n4 - Алгоритм PaCMAP.\n0 - Выход.\n\nВаш выбор: "))

    # Если выбор пользователя равен 1, то будет воспроизводиться алгоритм t-sne.

    # Как работает? Как работает: t-SNE старается разместить похожие точки 
    # (которые находятся рядом друг с другом в многомерном пространстве) рядом и в маленьком пространстве, но при этом может
    # "разбросать" те точки, которые далеко друг от друга

    if choose == 1:
        print("Преобразуем данные в массив NumPy")
        # Я везде поставил пока test_data из quick_test.py, там находится меньше значений (где-то 40 тысяч), посему визуализирование функционала
        # происходит быстрей, можете сменить на data, чтобы тестировать из полного списка mammoth_data.py.
        data_array = np.array(data_array_horse)

        print("Применение t-SNE для снижения размерности до 2D с прогресс-баром")

        # А теперь расскажу, что это за n_components, random_state и verbose, которые будут встречаться и дальше, а также чуть ниже
        
        # n_components. Это параметр, который говорит алгоритму, до скольких измерений нужно упростить данные.
        #  В твоем коде указано n_components=2. Это значит, что многомерные данные (которые могут иметь сотни измерений)
        #  преобразуются в 2 измерения, чтобы их можно было нарисовать на плоскости и посмотреть, как они выглядят. Если бы указали n_components=3, 
        # данные преобразовались бы в 3 измерения, и можно было бы создать 3D-график.

        # random_state — это как "фиксатор случайности", чтобы получить повторяемые результаты

        # verbose - Это настройка, которая определяет, будет ли программа показывать, как идет процесс работы 
        # (Иногда можно поставить 1 или True)
        tsne = TSNE(n_components=2, random_state=42, verbose=1)

        print("Применяем t-SNE")
        data_2d = tsne.fit_transform(data_array)

        # С этого момента происходит отрисовка графика с точками
        print("Визуализация результатов")
        plt.figure(figsize=(10, 8))

        print("Отрисовка всех точек")
        plt.scatter(data_2d[:, 0], data_2d[:, 1], s=10, c=data_2d[:, 0], cmap='viridis')

        plt.title('2D Visualization of Data using t-SNE')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid(True)
        plt.show()

    # Если выбор 2, то произойдёт анализ с помощью UMAP

    # Как работает? UMAP старается сохранить как локальные (близкие точки), так
    # и глобальные (далекие точки) связи между данными. Это значит, что он показывает не только маленькие группы данных,
    # но и общую картину.
    elif choose == 2:
        print("Преобразуем данные в массив NumPy")
        # Тут также, как и до этого. Использую test_data, вместо data. Если хотите ждать 3 часа, то пожалуйста, можете попробовать с data
        data_array = np.array(data_array_horse)

        print("Применение UMAP для снижения размерности до 2D")
        umap = UMAP(n_components=2, random_state=42, verbose=True)

        print("Применяем UMAP")
        data_2d = umap.fit_transform(data_array)

        # Визуализация с matplotlib
        print("Визуализация результатов")
        plt.figure(figsize=(10, 8))

        print("Отрисовка всех точек")
        plt.scatter(data_2d[:, 0], data_2d[:, 1], s=10, c=data_2d[:, 0], cmap='viridis')

        plt.title('2D Visualization of Data using UMAP')
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        plt.grid(True)
        plt.show()

    # Если выбор 3, то алгоритм TriMap

    # Как работает? TriMap сосредоточен на том, чтобы сохранить правильное расстояние между тройками точек. Это помогает
    # не только показать кластеры, но и передать, как они расположены друг относительно друга.
    elif choose == 3:
        print("Преобразуем данные в массив NumPy")
        data_array = np.array(data_array_horse)

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
                
    # Если выбор 4, то PaCMAP

    # CMAP пытается взять лучшее от всех предыдущих алгоритмов — он сохраняет как близкие точки, так и далекие,
    # показывая как мелкие детали, так и общую картину данных.
    elif choose == 4:
        print("Преобразуем данные в массив NumPy")
        data_array = np.array(data_array_horse)

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

    elif choose == 5:
        
        # Это больше тестирование, как работают вот эти данные с Лошадями
        # с алгоритмом тсне, МОЖНО НЕ ВКЛЮЧАТЬ, ЭТО ДЛЯ МЕНЯ
        
        # Сделал тут лошадинные колики, но только с t-SNE
        X, y, metadata = load_horse_colic_data()
        data_array = np.array(X)

        # Обработка NaN значений
        imputer = SimpleImputer(strategy='mean')
        data_array = imputer.fit_transform(data_array)

        # Кодирование целевых переменных
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y.values.flatten())  # Преобразуем в одномерный массив

        tsne = TSNE(n_components=2, random_state=42, verbose=1)
        data_2d = tsne.fit_transform(data_array)

        plt.figure(figsize=(10, 8))
        plt.scatter(data_2d[:, 0], data_2d[:, 1], s=10, c=y_encoded, cmap='viridis')
        plt.title('2D Visualization of Horse Colic Data using t-SNE')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid(True)
        plt.show()


    # Если 0, то выход из программы
    elif choose == 0:
        break

    # А это для особо отличившихся, кто ввел не те команды
    else:
        continue
