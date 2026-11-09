"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value
    result = 0
    vowels = ('a', 'e', 'i', 'o', 'u')
    for i in text.lower():
        if i in vowels:
            result += 1
    return result


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(set(text)) == len(text)


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    return bin(number).count('1')


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    count = 0
    while number >= 10:
        res = 1
        for i in str(number):
            res *= int(i)
        number = res
        count += 1
    return count


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    n = len(predicted)
    if n != len(expected):
        raise ValueError("Векторы должны быть одинаковой длины!")
    if n == 0:
        return 0.0
    sqr = 0
    for i in range(n):
        sqr += (predicted[i] - expected[i]) ** 2
    return sqr / n


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    result = ""
    d = 2
    while d * d <= number:
        f = 0
        while number % d == 0:
            number = number // d
            f += 1
        if f == 1:
            result += "(" + str(d) + ")"
        elif f > 1:
            result += "(" + str(d) + "**" + str(f) + ")"
        d += 1
    # остаток простой
    if number > 1:
        result += "(" + str(number) + ")"
    return result


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    k = 0
    while cube_count > 0:
        k += 1
        cube_count -= k ** 2
    if cube_count == 0 and k > 0:
        return k
    return "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    n = str(data.value)
    if len(n) % 2 == 0:
        mid = len(n) // 2 - 1
    else:
        mid = len(n) // 2
    left = 0
    right = 0
    for i in range(mid):
        left += int(n[i])
        right += int(n[-i - 1])
    return left == right


# тесты

def test(func, data, answer):
    result = func(data)
    print(f"{func.__name__}({data}) =", result)
    if result == answer:
        print("Тест успешно пройден!")
    else:
        print("Тест не пройден, ожидалось:", answer)
        raise AssertionError


if __name__ == "__main__":
    # Задача 1
    test(count_vowels, TextInput("data science"), 5)
    test(count_vowels, TextInput("AEIOU"), 5)
    test(count_vowels, TextInput("bcdfg"), 0)
    test(count_vowels, TextInput(""), 0)

    # Задача 2
    test(has_unique_characters, TextInput("abc"), True)
    test(has_unique_characters, TextInput("hello"), False)
    test(has_unique_characters, TextInput("aA"), True)
    test(has_unique_characters, TextInput(""), True)

    # Задача 3
    test(count_one_bits, PositiveIntegerInput(7), 3)
    test(count_one_bits, PositiveIntegerInput(1), 1)
    test(count_one_bits, PositiveIntegerInput(8), 1)
    test(count_one_bits, PositiveIntegerInput(255), 8)

    # Задача 4
    test(multiplicative_persistence, PositiveIntegerInput(39), 3)
    test(multiplicative_persistence, PositiveIntegerInput(4), 0)
    test(multiplicative_persistence, PositiveIntegerInput(999), 4)
    test(multiplicative_persistence, PositiveIntegerInput(10), 1)

    # Задача 5
    test(mse, VectorPairInput([1, 2, 3], [5, 6, 7]), 16.0)
    test(mse, VectorPairInput([0, 0, 0], [0, 0, 0]), 0.0)
    test(mse, VectorPairInput([1, 2], [2, 4]), 2.5)
    test(mse, VectorPairInput([-1], [1]), 4.0)

    # Задача 6
    test(prime_factorization, PositiveIntegerInput(86240), "(2**5)(5)(7**2)(11)")
    test(prime_factorization, PositiveIntegerInput(8), "(2**3)")
    test(prime_factorization, PositiveIntegerInput(2), "(2)")
    test(prime_factorization, PositiveIntegerInput(97), "(97)")
    test(prime_factorization, PositiveIntegerInput(1), "")

    # Задача 7
    test(pyramid, PositiveIntegerInput(1), 1)
    test(pyramid, PositiveIntegerInput(5), 2)
    test(pyramid, PositiveIntegerInput(14), 3)
    test(pyramid, PositiveIntegerInput(7), "It is impossible")
    test(pyramid, PositiveIntegerInput(2), "It is impossible")

    # Задача 8
    test(is_balanced_number, PositiveIntegerInput(1234006), True)
    test(is_balanced_number, PositiveIntegerInput(123456), False)
    test(is_balanced_number, PositiveIntegerInput(7), True)
    test(is_balanced_number, PositiveIntegerInput(12), True)
    test(is_balanced_number, PositiveIntegerInput(1230), False)
    test(is_balanced_number, PositiveIntegerInput(13722), True)

    print("\nВсе тесты пройдены!")
