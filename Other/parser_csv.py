import csv

# Функция для чтения CSV файла и преобразования в список
def csv_to_list(input_csv, output_py):
    data_list = []
    
    # Чтение данных из CSV файла
    with open(input_csv, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Пропускаем заголовки
        
        # Преобразуем строки CSV файла в списки
        for row in reader:
            data_list.append([float(x) for x in row])
    
    # Запись данных в Python файл в виде одного большого списка
    with open(output_py, 'w') as pyfile:
        pyfile.write("data = [\n")
        for item in data_list:
            pyfile.write(f"    {item},\n")
        pyfile.write("]\n")
    
    print(f"Данные успешно преобразованы и записаны в {output_py}")

# Укажите путь к входному CSV файлу и выходному .py файлу
csv_to_list('mammoth.csv', 'mammoth_data.py')
