from scipy.stats import rankdata
import numpy as np
from pathlib import Path
import argparse
from typing import NoReturn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str, default='in.txt',
                        help='Path to input file')
    parser.add_argument('-o', '--output_file', type=str, default='out.txt',
                        help='Path to output file')
    return parser.parse_args()


def readfile(filename: str) -> NoReturn:
    my_file = Path(filename)
    if my_file.is_file():
        with open(filename) as file:
            data = np.array([list(map(int, line.strip().split(' '))) for line in file.readlines()])
            if data.shape[1] != 2:
                raise ValueError("Data shape must be (N, 2)")
        return data
    else:
        raise FileNotFoundError(f'{filename} does not exist')


def writetofile(filename: str, data: tuple) -> NoReturn:
    output = str(data[0]) + ' ' + str(data[1]) + ' ' + str(data[2])
    with open(filename, 'w') as file:
        file.write(output)


def calculate(data: list) -> tuple:
    N = len(data)
    if N < 9:
        raise ValueError(f"Not enough data provided (N = {N}, must be at least 9)")

    data = sorted(data, key=lambda x: x[0])
    ranks = rankdata(data, axis=0)
    p = round(N / 3)

    r1 = np.sum(ranks[:p], axis=0)[1]
    r2 = np.sum(ranks[-p:], axis=0)[1]

    diff = round(r1 - r2)
    std = round((N + 0.5) * np.sqrt(p / 6))
    conj = round((r1 - r2) / (p * (N - p)), 2)

    return diff, std, conj


def main() -> NoReturn:
    args = parse_args()
    data = readfile(args.input_file)
    output = calculate(data)
    writetofile(args.output_file, output)


if __name__ == '__main__':
    main()
