import utils
import sys
import random


if __name__ == '__main__':
    PGM_NUM = 154
    pgms = [utils.read_PGM(f"../data/pgm/{i}.pgm") for i in range(1, PGM_NUM + 1)]

    S = int(sys.argv[1])  # number of substitutions

    for i in range(PGM_NUM):
        _, raw_x = pgms[i]

        flips = random.sample(range(len(raw_x)), S)
        prx = raw_x[:]
        for j in flips:
            prx[j] = 255 if prx[j] == 0 else 0  # flip

        utils.write_PGM(f"../data/random_binary_{S}/{i+1}.pgm", prx)
