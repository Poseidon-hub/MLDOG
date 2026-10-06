"""Пример: два имени одного списка и независимая копия."""


def main():
    original = [1, 2, 3]
    alias = original
    copied = original.copy()

    print("original is alias:", original is alias)
    print("original is copied:", original is copied)

    alias.append(4)
    print("После alias.append(4):", original, alias, copied)

    copied.append(5)
    print("После copied.append(5):", original, copied)


if __name__ == "__main__":
    main()
