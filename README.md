# Лабораторная работа №1 по обработке экспериментальных данных

## Описание
В данной лабораторной работе реализованы основные методы обработки экспериментальных данных на основе нормального распределения. Используются математические и статистические методы для анализа случайных величин, вычисления их характеристик, построения графиков и распределений.

## Установка и настройка проекта

1. Клонируйте репозиторий с проектом:
   ```bash
   git clone https://github.com/your-repo/lab-experimental-data.git
   ```

2. Перейдите в директорию проекта:
   ```bash
   cd lab-experimental-data
   ```

3. Создайте и активируйте виртуальное окружение (`venv`):

   - На Windows:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```

   - На Linux/MacOS:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

4. Установите необходимые зависимости:
   ```bash
   pip install -r requirements.txt
   ```

## Запуск проекта

1. Убедитесь, что виртуальное окружение активировано.
2. Запустите основной файл проекта:
   ```bash
   python Main.py
   ```

## Файлы

- `Main.py` — основной файл с кодом обработки данных и построения графиков.
- `requirements.txt` — список зависимостей для работы проекта.
- `.gitignore` — файл для исключения ненужных файлов из репозитория, включая `venv`.

## Используемые библиотеки

Проект использует следующие основные библиотеки:
- `numpy`
- `scipy`
- `matplotlib`
- `pandas`

### Примечание
Для работы с виртуальным окружением, всегда активируйте его перед запуском проекта.
