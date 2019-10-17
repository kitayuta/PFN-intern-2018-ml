import FGSM_utils
import utils
import sys
import random


if __name__ == '__main__':
    PGM_NUM = 154
    pgms = [utils.read_PGM(f"../data/pgm/{i}.pgm") for i in range(1, PGM_NUM + 1)]

    EPS = float(sys.argv[1])

    for i in range(PGM_NUM):
        _, raw_x = pgms[i]
        sign = random.choices([1, -1], k=len(raw_x))
        prx = FGSM_utils.perturb(raw_x, sign, EPS)

        utils.write_PGM(f"../data/random_{EPS:.3f}/{i+1}.pgm", prx)
