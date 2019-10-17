def sign(x):
    """Calculate the signs of a vector.

    Args:
        x (Vector): Input vector.

    Returns:
        list(int): Signs of the vector.

    """
    return [1 if x[i] > 0.0 else -1 for i in range(len(x))]


def clip(x):
    """Clip the value to [0, 255]. """
    return min(max(x, 0), 255)


def perturb(raw_x, s, eps):
    """Calculate the perturbed picture.

    Args:
        raw_x (list(int)): Raw pixel values of a picture.
        s (list(int)): List of the signs of the gradient.
        eps (float): \eps_0 of FGSM.

    Returns:
        list(int): Perturbed pixel values.

    """
    int_eps = int(eps * 255.0)
    px = [clip(raw_x[i] + (int_eps if s[i] > 0 else -int_eps)) for i in range(len(raw_x))]
    return px
