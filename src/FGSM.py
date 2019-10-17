import predictor
import FGSM_utils
import utils
import sys


if __name__ == '__main__':
    PGM_NUM = 154
    pgms = [utils.read_PGM(f"../data/pgm/{i}.pgm") for i in range(1, PGM_NUM + 1)]
    labels = utils.read_labels("../data/labels.txt", PGM_NUM)
    params = predictor.read_params("../data/param.txt")

    pred = predictor.Predictor(params)
    EPS = float(sys.argv[1])

    for i in range(PGM_NUM):
        x, raw_x = pgms[i]
        t = labels[i] - 1

        pred.forward(x)
        dx = pred.backward(t)  # get the gradient vector
        sign = FGSM_utils.sign(dx)
        prx = FGSM_utils.perturb(raw_x, sign, EPS)

        utils.write_PGM(f"../data/FGSM_{EPS:.3f}/{i+1}.pgm", prx)
