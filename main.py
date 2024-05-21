import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import fsolve
from functools import lru_cache
import FreeSimpleGUI as sg

# Параметры
alpha_range = np.linspace(0, 15, 100)  # Углы от 0 до 15 градусов
h0_values_default = [0.05, 0.1, 0.2, 0.5]  # Дефолтные значения для h0
h0_values = h0_values_default
current_profile = 1
first_profile = 1
second_profile = 2


# Определение функции формы и ее производной
def h_bar(x, h0, alpha, profile):
    if profile == 1:
        return h0 + alpha * x
    elif profile == 2:
        epsilon = 0.2
        return epsilon * x * (1 - x)


def h_bar_prime(x, alpha, profile):
    if profile == 1:
        return alpha
    elif profile == 2:
        epsilon = 0.2
        return epsilon * (1 - 2 * x)


# Кеширование этой функции, поскольку она вызывается повторно с одинаковыми аргументами
@lru_cache(maxsize=None)
def G_1(x, xi, h0, alpha, profile):
    epsilon = 1e-10  # Маленькое значение, чтобы избежать деления на ноль
    h_b = h_bar(x, h0, alpha, profile)
    h_b_prime = h_bar_prime(x, alpha, profile)
    # Добавление эпсилона к знаменателям, чтобы избежать деления на ноль
    term1 = 16 * h_b ** 2 / (x - xi + epsilon) / ((x - xi) ** 2 + 16 * h_b ** 2 + epsilon)
    term2 = -8 * h_b * h_b_prime * ((x - xi) ** 2 - 16 * h_b ** 2) / abs((x - xi) ** 2 + 16 * h_b ** 2 + epsilon)
    return term1 + term2


@lru_cache(maxsize=None)
def f1_x(x, h0, alpha, profile):
    integrand = lambda xi: np.sqrt((1 + xi) / (1 - xi)) * G_1(x, xi, h0, alpha, profile)
    return quad(integrand, -1, 1, epsabs=1e-12, epsrel=1e-12, limit=200)[0] / (2 * np.pi)


@lru_cache(maxsize=None)
def phi1_x(x, h0, alpha, profile):
    integrand = lambda xi: np.sqrt(1 - xi ** 2) * G_1(x, xi, h0, alpha, profile)
    return quad(integrand, -1, 1, epsabs=1e-12, epsrel=1e-12, limit=200)[0] / (2 * np.pi)


@lru_cache(maxsize=None)
def f2_x(x, h0, alpha, profile):
    epsilon = 1e-10  # Маленькое значение, чтобы избежать деления на ноль
    integrand = lambda s: f1_x(s, h0, alpha, profile) / (2 * h_bar(s, h0, alpha, profile) + epsilon)
    return quad(integrand, -1, x, epsabs=1e-12, epsrel=1e-12, limit=200)[0]


@lru_cache(maxsize=None)
def phi2_x(x, h0, alpha, profile):
    epsilon = 1e-10  # Маленькое значение, чтобы избежать деления на ноль
    integrand = lambda s: phi1_x(s, h0, alpha, profile) / (2 * h_bar(s, h0, alpha, profile) + epsilon)
    return quad(integrand, -1, x, epsabs=1e-12, epsrel=1e-12, limit=200)[0]


# Решение для коэффициентов A и B
def solve_A_B(h0, alpha, profile):
    def equations(p):
        A, B = p
        return (
            4 * h_bar(1, h0, alpha, profile) - 4 * h_bar(-1, h0, alpha, profile) + B * phi2_x(1, h0, alpha,
                                                                                              profile) + A * f2_x(1, h0,
                                                                                                                  alpha,
                                                                                                                  profile),
            8 * h_bar(-1, h0, alpha, profile) * h_bar_prime(-1, alpha, profile) + A * f1_x(1, h0, alpha,
                                                                                           profile) + B * phi1_x(-1, h0,
                                                                                                                 alpha,
                                                                                                                 profile))

    A, B = fsolve(equations, (0, 0))
    return A, B


def W_0(x, h0, alpha, profile):
    h_b = h_bar(x, h0, alpha, profile)
    return 1 / np.pi * (np.arctan((1 + x) / (4 * h_b)) + np.arctan((1 - x) / (4 * h_b)))


def r(x, h0, alpha, A, B, profile):
    return (4 * h_bar(x, h0, alpha, profile) - 4 * h_bar(-1, h0, alpha, profile) + A * f2_x(x, h0, alpha,
                                                                                            profile) + B * phi2_x(x, h0,
                                                                                                                  alpha,
                                                                                                                  profile)) * W_0(
        x, h0, alpha, profile)


def gamma1(x, h0, alpha, A, B, profile):
    return (A * np.sqrt((1 + x) / (1 - x)) + B * np.sqrt(1 - x ** 2) + r(x, h0, alpha, A, B, profile)) / (
            4 * h_bar(x, h0, alpha, profile))


# Расчет коэффициента подъемной силы
def calculate_cy(h0, alpha, profile):
    A, B = solve_A_B(h0, alpha, profile)
    integrand = lambda x: gamma1(x, h0, alpha, A, B)
    return quad(integrand, -1, 1, epsabs=1e-12, epsrel=1e-12, limit=200)[0]


@lru_cache(maxsize=None)
def phi_x(x, y, h0, alpha, A, B, profile):
    epsilon = 4 * h_bar(x, h0, alpha, profile)
    integrand = lambda xi: gamma1(xi, h0, alpha, A, B) * (
            (y - h_bar(xi, h0, alpha, profile)) / ((x - xi) ** 2 + (y - h_bar(xi, h0, alpha, profile)) ** 2 + epsilon) -
            (y + h_bar(xi, h0, alpha, profile) + epsilon) / (
                    (x - xi) ** 2 + (y + h_bar(xi, h0, alpha, profile) + epsilon) ** 2)
    )
    return -1 / (2 * np.pi) * quad(integrand, -1, 1, epsabs=1e-12, epsrel=1e-12, limit=200)[0]


@lru_cache(maxsize=None)
def phi_y(x, y, h0, alpha, A, B, profile):
    epsilon = 4 * h_bar(x, h0, alpha, profile)
    integrand = lambda xi: gamma1(xi, h0, alpha, A, B) * (
            (x - xi) / ((x - xi) ** 2 + (y - h_bar(xi, h0, alpha, profile)) ** 2 + epsilon) -
            (x - xi) / ((x - xi) ** 2 + (y + h_bar(xi, h0, alpha, profile) + epsilon) ** 2)
    )
    return 1 / (2 * np.pi) * quad(integrand, -1, 1, epsabs=1e-12, epsrel=1e-12, limit=200)[0]


def Cp(x, gamma, phi_x, phi_y):
    gamma_squared = gamma ** 2
    phi_x_squared = phi_x ** 2
    phi_y_squared = phi_y ** 2
    return -(gamma_squared / 8) + phi_x - (phi_x_squared + phi_y_squared) / 2 - (gamma * (1 - phi_x)) / 2


# Функция для расчета x-координаты центра давления (X_alpha)
def calculate_X_alpha(h0, alpha, profile):
    A, B = solve_A_B(h0, alpha, profile)
    # Устанавливаем y равным нулю, так как нас интересует распределение вдоль хорды для 2D сечения
    y = 0
    numerator = lambda x: x * Cp(x, gamma1(x, h0, alpha, A, B, profile), phi_x(x, y, h0, alpha, A, B, profile),
                                 phi_y(x, y, h0, alpha, A, B, profile))
    numerator_result, _ = quad(numerator, -1, 1, epsabs=1e-12, epsrel=1e-12, limit=200)

    denominator = lambda x: Cp(x, gamma1(x, h0, alpha, A, B, profile), phi_x(x, y, h0, alpha, A, B, profile),
                               phi_y(x, y, h0, alpha, A, B, profile))
    denominator_result, _ = quad(denominator, -1, 1, epsabs=1e-12, epsrel=1e-12, limit=200)

    # Проверка для избежания деления на ноль
    if denominator_result != 0:
        X_alpha = numerator_result / denominator_result
    else:
        X_alpha = np.nan  # NaN указывает на неопределенный результат

    return X_alpha


def calculate_lifting_force():
    # Подготовить plot
    plt.figure(figsize=(10, 6))

    # Вычисляем Cy для каждого значения h0
    for h0 in h0_values:
        cy_values = [calculate_cy(h0, alpha, current_profile) for alpha in alpha_range]
        plt.plot(alpha_range, cy_values, label=f'h0={h0}')

    # Форматируем наш plot
    plt.title('Коэффициент Подъемной Силы')
    plt.xlabel('$\\alpha$, град')
    plt.ylabel('$C_y$')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 15)
    plt.ylim(0, 2)

    # Показать plot
    plt.show()


def calculate_pressure_center():
    # Подготовить plot
    plt.figure(figsize=(10, 6))

    # Вычисляем центр давлений для каждого значения h0
    for h0 in h0_values:
        x_alpha_values = [calculate_X_alpha(h0, alpha, current_profile) for alpha in alpha_range]
        plt.plot(alpha_range, x_alpha_values, label=f'h0={h0}')

    # Форматируем наш plot
    plt.title('Положение центра давлений относительно носика профиля')
    plt.xlabel('$\\alpha$, град')
    plt.ylabel('$X_$\\alpha$$')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 15)
    plt.ylim(0, 2)

    # Показать plot
    plt.show()


def compare_lifting_force():
    # Подготовить plot
    plt.figure(figsize=(10, 6))

    # Вычисляем Cy для двух профилей
    x_alpha_values = [calculate_X_alpha(h0_values, alpha, first_profile) for alpha in alpha_range]
    plt.plot(alpha_range, x_alpha_values, label=f'h0={h0_values}')
    x_alpha_values = [calculate_X_alpha(h0_values, alpha, second_profile) for alpha in alpha_range]
    plt.plot(alpha_range, x_alpha_values, label=f'h0={h0_values}')

    # Форматируем наш plot
    plt.title('Коэффициент Подъемной Силы')
    plt.xlabel('$\\alpha$, град')
    plt.ylabel('$X_$\\alpha$$')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 15)
    plt.ylim(0, 2)

    # Показать plot
    plt.show()


def compare_pressure_center():
    # Подготовить plot
    plt.figure(figsize=(10, 6))

    # Вычисляем центр давлений для двух профилей
    x_alpha_values = [calculate_X_alpha(h0_values, alpha, first_profile) for alpha in alpha_range]
    plt.plot(alpha_range, x_alpha_values, label=f'h0={h0_values}')
    x_alpha_values = [calculate_X_alpha(h0_values, alpha, second_profile) for alpha in alpha_range]
    plt.plot(alpha_range, x_alpha_values, label=f'h0={h0_values}')

    # Форматируем наш plot
    plt.title('Положение центра давлений относительно носика профиля')
    plt.xlabel('$\\alpha$, град')
    plt.ylabel('$X_$\\alpha$$')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 15)
    plt.ylim(0, 2)

    # Показать plot
    plt.show()


profiles = ['Flat plate', 'Parabolic curve']
characteristics = ['Lifting force', 'Pressure center']

# Каркас приложения для построения графиков
layout_disp = [[sg.Text('Display graph', font='bold')],
               [sg.Text('Choose a profile')],
               [sg.Combo(profiles, key='profile1', default_value=profiles[0])],
               [sg.Text('Choose the characteristic')],
               [sg.Combo(characteristics, key='characteristic1', default_value=characteristics[0])],
               [sg.Text('Enter the relative distances separated by a space')],
               [sg.InputText(default_text='0.05 0.1 0.2 0.5')],
               [sg.Text('', key='error1', text_color='orange', font='bold')],
               [sg.Button('Display')]]

# Каркас приложения для отображения сравнения графиков
layout_comp = [[sg.Text('Compare data', font='bold')],
               [sg.Text('Choose a profile')],
               [sg.Combo(profiles, key='profile2', default_value=profiles[0]),
                sg.Combo(profiles, key='profile3', default_value=profiles[1])],
               [sg.Text('Choose the characteristic')],
               [sg.Combo(characteristics, key='characteristic2', default_value=characteristics[0])],
               [sg.Text('Choose the relative distance')],
               [sg.Combo(h0_values_default, key='h0', default_value=h0_values_default[0])],
               [sg.Text('', key='error2', text_color='orange', font='bold')],
               [sg.Button('Compare')]]

layout = [[sg.Column(layout_disp), sg.VerticalSeparator(), sg.Column(layout_comp)]]

# Создаем окно
window = sg.Window('Aero+', layout)

# Цикл событий для обработки «событий» и получения «значений» входных данных.
while True:
    event, values = window.read()

    # если пользователь закрывает окно или нажимает кнопку «Отмена»
    if event == sg.WIN_CLOSED or event == 'Cancel':
        break
    elif event == 'Display':
        if values['profile1'] == 'Flat plate':
            current_profile = 1
        elif values['profile1'] == 'Parabolic curve':
            current_profile = 2

        try:
            h0_values = list(map(float, layout_disp[5][0].get().split()))
        except:
            window['error1'].update(value='Incorrect input!')

        if values['characteristic1'] == 'Lifting force':
            calculate_lifting_force()
        elif values['characteristic1'] == 'Pressure center':
            calculate_pressure_center()

        h0_values = h0_values_default

    elif event == 'Compare':
        if values['profile2'] == values['profile3']:
            window['error2'].update(value='Profiles identical!')
        else:
            first_profile = profiles.index(values['profile2']) + 1
            second_profile = profiles.index(values['profile3']) + 1

            h0_values = values['h0']

            if values['characteristic2'] == 'Lifting force':
                compare_lifting_force()
            elif values['characteristic2'] == 'Pressure center':
                compare_pressure_center()

            h0_values = h0_values_default

window.close()
