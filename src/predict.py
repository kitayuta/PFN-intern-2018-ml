import sys
import vector
import predictor
import utils

if __name__ == '__main__':
    PGM_NUM = 154
    pgms = [utils.read_PGM(f"../data/{sys.argv[1]}/{i}.pgm") for i in range(1, PGM_NUM + 1)]
    labels = utils.read_labels("../data/labels.txt", PGM_NUM)
    params = predictor.read_params("../data/param.txt")

    pred = predictor.Predictor(params)

    count = 0
    for i in range(PGM_NUM):
        x, _ = pgms[i]
        ps = pred.forward(x)
        l = vector.argmax(ps) + 1
        print(f"{i+1}.pgm: predicted = {l} (true = {labels[i]})")
        if l == labels[i]:
            count += 1

    print()
    print(f"accuracy = {count / PGM_NUM}")
