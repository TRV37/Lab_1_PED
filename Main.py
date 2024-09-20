import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# Заданные параметры
a = 5
sigma = 2
x = 3
F0 = 0.92
t0 = 5
x1 = 3
x2 = 8

# а) Значение функции плотности и функции распределения в точке x
f_x = stats.norm.pdf(x, loc=a, scale=sigma)
F_x = stats.norm.cdf(x, loc=a, scale=sigma)

# б) Значение аргумента x0, при котором функция распределения примет значение F0
x0 = stats.norm.ppf(F0, loc=a, scale=sigma)

# в) Значение параметра t для стандартного нормального распределения
t = (x - a) / sigma

# г) Значение функции плотности и функции распределения для t в стандартном нормальном распределении
f_n_t = stats.norm.pdf(t)
F_n_t = stats.norm.cdf(t)

# д) Значение функции Лапласа в точке t0 (функция распределения стандартного нормального распределения)
laplace_t0 = stats.norm.cdf(t0)

# e) Вероятность попадания значения случайной величины в интервал [x1, x2]
probability = stats.norm.cdf(x2, loc=a, scale=sigma) - stats.norm.cdf(x1, loc=a, scale=sigma)

# Шаг 2: построение таблицы значений функции плотности на интервале с шагом 0.5
x_values = np.arange(a - 3*sigma - 1, a + 3*sigma + 1, 0.5)
f_values = stats.norm.pdf(x_values, loc=a, scale=sigma)

# Создание таблицы с помощью pandas
df = pd.DataFrame({
    'x': x_values,
    'f(x)': f_values
})

# Вывод таблицы
print(df)

# Генерация выборки 50 случайных чисел
sample = np.random.normal(loc=a, scale=sigma, size=50)

# Расчет количества и относительной частоты попаданий случайных чисел в интервалы
bin_counts, bin_edges = np.histogram(sample, bins=np.arange(a - 3*sigma - 1, a + 3*sigma + 1, 0.5))
relative_frequencies = bin_counts / len(sample)

# Построение комбинированной диаграммы
plt.figure(figsize=(10, 6))
plt.bar(bin_edges[:-1], relative_frequencies, width=0.5, alpha=0.6, label='Relative Frequencies')
plt.plot(x_values, f_values, color='red', label='PDF', linewidth=2)
plt.title('Combined Histogram and PDF')
plt.xlabel('x')
plt.ylabel('Density / Relative Frequency')
plt.legend()
plt.grid(True)
plt.show()

# Построение графиков функции плотности и функции распределения
x_range = np.linspace(a - 4*sigma, a + 4*sigma, 1000)
pdf_values = stats.norm.pdf(x_range, loc=a, scale=sigma)
cdf_values = stats.norm.cdf(x_range, loc=a, scale=sigma)

# Создание графиков
plt.figure(figsize=(10, 6))

# График функции плотности
plt.subplot(2, 1, 1)
plt.plot(x_range, pdf_values, label="PDF", color="blue")
plt.title("Normal Distribution PDF and CDF")
plt.ylabel("Probability Density")
plt.grid(True)

# График функции распределения
plt.subplot(2, 1, 2)
plt.plot(x_range, cdf_values, label="CDF", color="green")
plt.ylabel("Cumulative Probability")
plt.xlabel("x")
plt.grid(True)

plt.tight_layout()
plt.show()

print(f_x, F_x, x0, t, f_n_t, F_n_t, laplace_t0, probability)
