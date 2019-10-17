import vector
import os


def read_PGM(filename):
    """Read a PGM file.

    Args:
        filename (str): Filename of a PGM file.

    Returns:
        (Vector, list(int)): Tuple of (normalized vector of the pixels,
            list of raw pixel values).

    """
    with open(filename) as f:
        for _ in range(3):
            f.readline()  # discard

        pixels, raw_pixels = [], []
        for _ in range(32):
            line = f.readline()
            vs = [int(s) for s in line.split()]
            pixels.extend([v / 255.0 for v in vs])
            raw_pixels.extend(vs)

    assert len(pixels) == 1024  # validate

    return vector.Vector(pixels), raw_pixels


def write_PGM(filename, raw_pixels):
    """Write a PGM file.

    Args:
        filename (str): Filename of a PGM file.
        raw_pixels (list(int)): Raw pixel values.

    """
    assert len(raw_pixels) == 1024  # validate

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as f:
        f.write("P2\n")
        f.write("32 32\n")
        f.write("255")
        for i, p in enumerate(raw_pixels):
            if i % 32 == 0:
                f.write("\n")
            else:
                f.write(" ")
            f.write(str(p))


def read_labels(filename, n):
    """Read labels.

    Args:
        filename (str): Filename of the labels.
        n (int): Number of the labels.

    Returns:
        list(int): List of the labels.

    """
    with open(filename) as f:
        ls = []
        for _ in range(n):
            ls.append(int(f.readline()))
        return ls
