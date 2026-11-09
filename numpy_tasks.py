"""Заготовки задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    result = np.zeros(np.shape(vectors[0]))
    for i in range(len(matrices)):
        result = result + np.dot(matrices[i], vectors[i])
    return result


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    binary_M = np.where(np.asarray(matrix) > threshold, 1, 0)
    return binary_M


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    unique_rows = []
    for row in matrix:
        unique_rows.append(np.unique(row).tolist())
    return unique_rows


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    unique_cols = []
    for i in range(matrix.shape[1]):
        unique_cols.append(np.unique(matrix[:, i]).tolist())
    return unique_cols


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    rng = np.random.default_rng(seed)
    mat = rng.normal(mean, std, size=(rows, columns))
    row_mat = np.mean(mat, axis=1)
    col_mat = np.mean(mat, axis=0)
    row_v = np.var(mat, axis=1)
    col_v = np.var(mat, axis=0)
    return MatrixStatistics(mat, row_mat, col_mat, row_v, col_v)


def plot_histograms(mat):
    import matplotlib.pyplot as plt
    m, n = mat.shape

    fig, axes = plt.subplots(1, m, figsize=(3 * m, 3))
    for i in range(m):
        ax = axes[i] if m > 1 else axes
        ax.hist(mat[i, :], bins=max(n // 2, 1), color="b", edgecolor="black")
        ax.set_title(f"Row {i + 1}")
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    for i in range(n):
        ax = axes[i] if n > 1 else axes
        ax.hist(mat[:, i], bins=max(m // 2, 1), color="g", edgecolor="black")
        ax.set_title(f"Column {i + 1}")
    plt.tight_layout()
    plt.show()


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    M = np.indices((rows, columns)).sum(axis=0) % 2
    M = np.where(M == 0, first, second)
    return M


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    img = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    img[:, :] = background_color
    top = (image_height - height) // 2
    left = (image_width - width) // 2
    top, left = max(top, 0), max(left, 0)
    img[top:top + height, left:left + width] = shape_color
    return img


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    img = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    img[:, :] = background_color
    x0 = (image_width - 1) / 2
    y0 = (image_height - 1) / 2
    y, x = np.indices((image_height, image_width))
    mask = (x - x0) ** 2 / semi_axis_x ** 2 + (y - y0) ** 2 / semi_axis_y ** 2 <= 1
    img[mask] = shape_color
    return img


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = np.asarray(data.values, dtype=float), data.window
    mat = float(np.mean(values))
    var = float(np.var(values))
    std = float(np.std(values))
    local_min = []
    local_max = []
    for i in range(1, len(values) - 1):
        if values[i] < values[i - 1] and values[i] < values[i + 1]:
            local_min.append(i)
        if values[i] > values[i - 1] and values[i] > values[i + 1]:
            local_max.append(i)
    method = np.convolve(values, np.ones(window), mode="valid") / window
    return TimeSeriesStatistics(mat, var, std, local_max, local_min, method)


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = np.asarray(data.labels, dtype=int), data.class_count
    if class_count is None:
        class_count = labels.max() + 1 if len(labels) > 0 else 0
    encod = np.zeros((len(labels), class_count), dtype=int)
    for i, label in enumerate(labels):
        encod[i, label] = 1
    return encod


# тесты

def test(name, result, answer):
    print(name)
    if np.array_equal(np.asarray(result), np.asarray(answer)):
        print("Тест успешно пройден!")
    else:
        print("Тест не пройден")
        print("Получили:", result)
        print("Ожидали:", answer)
        raise AssertionError


if __name__ == "__main__":
    # Задача 1
    X = [np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])]
    V = [np.array([[1], [2], [3]]), np.array([[3], [2], [1]])]
    test("sum_prod", sum_prod(MatrixVectorBatchInput(X, V)), [[64], [64], [64]])
    test("sum_prod одна матрица", sum_prod(MatrixVectorBatchInput([np.eye(2)], [np.array([[5], [7]])])), [[5], [7]])

    # Задача 2
    M = np.array([[0.1, 0.5, 0.9], [1.0, 0.0, 0.6]])
    test("binarize", binarize(BinarizeInput(M)), [[0, 0, 1], [1, 0, 1]])
    test("binarize threshold=0", binarize(BinarizeInput(M, 0)), [[1, 1, 1], [1, 0, 1]])
    test("binarize исходная не изменилась", M, [[0.1, 0.5, 0.9], [1.0, 0.0, 0.6]])

    # Задача 3
    m_rows = [[1, 2, 3, 4, 5], [5, 6, 7, 7, 4], [5, 4, 3, 1, 3]]
    print(unique_rows(MatrixInput(m_rows)))
    assert unique_rows(MatrixInput(m_rows)) == [[1, 2, 3, 4, 5], [4, 5, 6, 7], [1, 3, 4, 5]]
    m_cols = [[1, 5], [1, 4], [2, 5]]
    print(unique_columns(MatrixInput(m_cols)))
    assert unique_columns(MatrixInput(m_cols)) == [[1, 2], [4, 5]]
    assert unique_rows(MatrixInput([[7, 7, 7]])) == [[7]]

    # Задача 4
    st = matrix_statistics(RandomMatrixInput(5, 3, mean=10, std=2, seed=1))
    print("Матрица:\n", st.matrix)
    print("Мат.ожидания для строк:", st.row_means)
    print("Мат.ожидания для столбцов:", st.column_means)
    print("Дисперсии для строк:", st.row_variances)
    print("Дисперсии для столбцов:", st.column_variances)
    assert st.matrix.shape == (5, 3)
    assert np.allclose(st.row_means, st.matrix.mean(axis=1))
    st2 = matrix_statistics(RandomMatrixInput(5, 3, mean=10, std=2, seed=1))
    test("matrix_statistics одинаковый seed", st.matrix, st2.matrix)
    big = matrix_statistics(RandomMatrixInput(300, 300, mean=10, std=2))
    assert abs(big.matrix.mean() - 10) < 0.1 and abs(big.matrix.std() - 2) < 0.1

    # Задача 5
    test("chess", chess(ChessInput(3, 4, 0, 1)), [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
    test("chess 1x1", chess(ChessInput(1, 1, 5, 9)), [[5]])
    test("chess 2x2", chess(ChessInput(2, 2, 5, 9)), [[5, 9], [9, 5]])

    # Задача 6
    rect = draw_rectangle(RectangleInput(2, 2, 4, 4, (255, 0, 0), (0, 0, 0)))
    assert rect.shape == (4, 4, 3)
    test("draw_rectangle центр", rect[1:3, 1:3], np.full((2, 2, 3), [255, 0, 0]))
    test("draw_rectangle угол", rect[0, 0], [0, 0, 0])
    ell = draw_ellipse(EllipseInput(3, 2, 7, 9, (0, 255, 0), (255, 255, 255)))
    assert ell.shape == (7, 9, 3)
    test("draw_ellipse центр", ell[3, 4], [0, 255, 0])
    test("draw_ellipse угол", ell[0, 0], [255, 255, 255])
    test("draw_ellipse край по x", ell[3, 1], [0, 255, 0])
    test("draw_ellipse за краем по x", ell[3, 0], [255, 255, 255])

    # Задача 7
    ts = np.array([7, 9, 10, 11, 6, 4, 1, 3, 5, 2])
    res = analyze_time_series(TimeSeriesInput(ts, 4))
    print("Мат.ожидание:", res.mean)
    print("Дисперсия:", res.variance)
    print("СКО:", res.std)
    print("Локальные максимумы:", res.local_maxima_indices)
    print("Локальные минимумы:", res.local_minima_indices)
    print("Метод скользящего среднего:", res.moving_average)
    test("локальные максимумы", res.local_maxima_indices, [3, 8])
    test("локальные минимумы", res.local_minima_indices, [6])
    test("скользящее среднее", res.moving_average, [9.25, 9, 7.75, 5.5, 3.5, 3.25, 2.75])
    test("ровный ряд без экстремумов", analyze_time_series(TimeSeriesInput([1, 1, 1], 1)).local_maxima_indices, [])

    # Задача 8
    test("one_hot", one_hot(OneHotInput([0, 2, 3, 0])), [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]])
    test("one_hot class_count", one_hot(OneHotInput([1, 0], class_count=4)), [[0, 1, 0, 0], [1, 0, 0, 0]])
    test("one_hot одна метка", one_hot(OneHotInput([5])), [[0, 0, 0, 0, 0, 1]])

    print("\nВсе тесты пройдены!")

    # гистограммы к задаче 4
    plot_histograms(st.matrix)
