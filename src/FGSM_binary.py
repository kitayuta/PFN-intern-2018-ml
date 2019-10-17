import predictor
import utils
import sys


if __name__ == '__main__':
    PGM_NUM = 154
    pgms = [utils.read_PGM(f"../data/pgm/{i}.pgm") for i in range(1, PGM_NUM + 1)]
    labels = utils.read_labels("../data/labels.txt", PGM_NUM)
    params = predictor.read_params("../data/param.txt")

    pred = predictor.Predictor(params)

    S = int(sys.argv[1])  # number of substitutions

    for i in range(PGM_NUM):
        x, raw_x = pgms[i]
        t = labels[i] - 1

        pred.forward(x)
        dx = pred.backward(t)  # get the gradient vector
        # Calculate the rate of change of the loss function by flipping each pixel
        dx_flip = [dx[j] if raw_x[j] == 0 else -dx[j] for j in range(len(x))]

        # Get the top S pixels
        tops = sorted([(dx, j) for (j, dx) in enumerate(dx_flip)], reverse=True)[:S]
        prx = raw_x[:]
        for _, j in tops:
            prx[j] = 255 if prx[j] == 0 else 0  # flip the top S pixels

        utils.write_PGM(f"../data/FGSM_binary_{S}/{i+1}.pgm", prx)
