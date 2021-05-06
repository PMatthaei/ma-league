import numpy as np
import multiprocessing as mp


class Tester:
    num = 0.0
    name = 'none'

    def __init__(self, tnum=num, tname=name):
        self.num = tnum
        self.name = tname

    def __str__(self):
        return '%f %s' % (self.num, self.name)


def mod(test, nn, out_queue):
    print(test.num)
    np.random.seed(int(test.num))
    test.num = np.random.randn()
    print(test.num)
    test.name = nn
    out_queue.put(test)


if __name__ == '__main__':
    num = 10
    out_queue = mp.Queue()
    tests = np.empty(num, dtype=object)
    for it in range(num):
        tests[it] = Tester(tnum=it * 1.0)

    print('\n')
    workers = [mp.Process(target=mod, args=(test, 'some', out_queue)) for test in tests]

    for work in workers: work.start()

    for work in workers: work.join()

    res_lst = []
    for j in range(len(workers)):
        res_lst.append(out_queue.get())

    print(res_lst)
