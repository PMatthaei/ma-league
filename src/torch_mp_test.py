import sys
import time

import torch
from torch.multiprocessing import Process as TorchProcess
from torch.multiprocessing import Queue as TorchQueue


def sample_data():
    return torch.rand([1000, 128, 72, 3], dtype=torch.float)


def torch_shared_mem_process(in_q: TorchQueue, out_q: TorchQueue):
    counter = 0

    while True:
        data = in_q.get()
        counter += 1
        data *= 1.001
        out_q.put(data)
        if data is None:
            return


def test_mem_share():
    ps = []
    in_qs = []
    out_qs = []
    for i in range(15):
        in_q = TorchQueue()
        out_q = TorchQueue()
        p = TorchProcess(target=torch_shared_mem_process, args=(in_q, out_q, ))
        ps.append(p)
        in_qs.append(in_q)
        out_qs.append(out_q)
        p.start()

    start = time.time()

    n = 500
    for i in range(n):
        data = sample_data()

        [q.put(data) for q in in_qs]
        gets = [q.get() for q in out_qs]

        print(f'Progress {i}/{n}')

    [q.put(None) for q in in_qs]
    [p.join() for p in ps]

    print(f'Finished sending {n} tensor lists!')

    took_seconds = time.time() - start
    return took_seconds


def main():
    with_shared_memory = test_mem_share()

    print(f'Took {with_shared_memory:.1f} s with shared memory.')


if __name__ == '__main__':
    sys.exit(main())
