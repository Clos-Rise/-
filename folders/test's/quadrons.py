import numpy as np
import pickle
from numba import cuda

def generate(limit=500):
    quadrons = []
    for i in range(limit):
        quadron = np.random.rand(4, 4, 4, 4).astype(np.float32)
        quadrons.append(quadron)
    return quadrons

def save_quadrons(quadrons, filename):
    try:
        with open(filename, 'wb') as f:
            pickle.dump(quadrons, f)
    except Exception as e:
        print(f"Ошибка сохранения: {e}")

@cuda.jit
def proc_quadrons(quadrons, result):
    idx = cuda.grid(1)
    if idx < quadrons.shape[0]:
        quadron = quadrons[idx]
        sum_val = 0.0
        for i in range(quadron.shape[0]):
            for j in range(quadron.shape[1]):
                for k in range(quadron.shape[2]):
                    for l in range(quadron.shape[3]):
                        sum_val += quadron[i, j, k, l]
        result[idx] = sum_val

def main():
    try:
        quadrons = generate()

        quadrons_a = np.array(quadrons, dtype=np.float32)

        result = np.zeros(len(quadrons), dtype=np.float32)

        quadrons_d = cuda.to_device(quadrons_a)
        result = cuda.to_device(result)

        tpb = 256
        blocks = (quadrons_a.shape[0] + (tpb - 1)) // tpb
        proc_quadrons[blocks, tpb](quadrons_d, result)

        results = result.copy_to_host()

        save_quadrons(quadrons, 'main.quadrons')

        print("Вывод:", results)
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    main()
