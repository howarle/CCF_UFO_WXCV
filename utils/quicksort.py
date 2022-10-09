from dataclasses import is_dataclass
from math import log
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import random

class quicksort:
    def __init__(self, a, minimum_size=200) -> None:
        self.ans = None
        self.recod = {}
        self.is_done = False
        self.tot_size = len(a)
        self.minimum_size = minimum_size
        # -------
        random.shuffle(a)
        # -------
        
        self.seg_list = [qiuckSegment(a, self.minimum_size)]

    def query(self) -> list:
        
        print("querying...", end="")
        ask_list = [] 
        # for it in self.seg_list:
        #     ask_list = ask_list + it.query()
        with ThreadPoolExecutor(20) as executor:
            th_list = []
            for it in self.seg_list:
                th_list.append(executor.submit(it.query))
            for th in th_list:
                ask_list = ask_list + th.result()
        
        print("query done")
        return ask_list

    def do(self, recod) -> bool:
        if self.is_done:
            return True
        def get_recod(x, y):
            return recod[tuple((x, y))] if x < y else -recod[tuple((y, x))]

        cnt = 0
        seg_list = []
        # for it in self.seg_list:
        #     if it.do(recod):
        #         cnt = cnt + 1
        #         if len(it.ans[0]) > 0:
        #             mg1 = qiuckSegment(it.ans[0], self.minimum_size)
        #             seg_list.append(mg1)
        #         if len(it.ans[1]) > 0:
        #             mg2 = qiuckSegment(it.ans[1], self.minimum_size)
        #             seg_list.append(mg2)
        #     else:
        #         seg_list.append(it)

        print("doing...", end="")
        with ThreadPoolExecutor(20) as executor:
            th_list = []
            for it in self.seg_list:
                th_list.append(executor.submit(it.do, recod))

            for idx, th in enumerate(th_list):
                it = self.seg_list[idx]
                if th.result():
                    cnt = cnt + 1
                    if len(it.ans[0]) > 0:
                        mg1 = qiuckSegment(it.ans[0], self.minimum_size)
                        seg_list.append(mg1)
                    if len(it.ans[1]) > 0:
                        mg2 = qiuckSegment(it.ans[1], self.minimum_size)
                        seg_list.append(mg2)
                else:
                    seg_list.append(it)
        print("done")
        self.seg_list = seg_list

        if cnt == 0:
            self.ans = []
            for it in self.seg_list:
                assert(len(it.ans) == 1)
                self.ans = self.ans + it.ans[0]
            self.is_done = True
        return self.is_done


class qiuckSegment:
    def __init__(self, a: list, minimum_size) -> None:
        self.minimum_siz = minimum_size
        self.tot_siz = len(a)
        self.ans = None
        self.bucket = []
        self.is_done = False
        self.a = a
        self.need_split = False
        self.bucket_siz = max(10, 10+int(log(self.tot_siz, 2)))
        # self.bucket_siz = max(10, int(self.tot_siz*0.001))

    def __add_ask(self, x, y) -> None:
        if x > y:
            x, y = y, x
        self.ask_list.append(tuple((x, y)))

    def query(self) -> list:
        if self.is_done:
            return []
        ask_list = []
        def add_ask(x, y) -> None:
            if x > y:
                x, y = y, x
            ask_list.append(tuple((x, y)))
        
        if self.tot_siz <= self.minimum_siz:
            for i1 in range(len(self.a)-1):
                for i2 in range(i1+1, len(self.a)):
                    add_ask(self.a[i1], self.a[i2])
        else:
            self.bucket = random.sample(self.a, self.bucket_siz)
            for x in self.a:
                for y in  self.bucket:
                    add_ask(x, y)

        return ask_list

    def do(self, recod) -> bool:
        '''
            return: True if need split
        '''
        if self.is_done:
            return False
        def get_recod(x, y):
            return recod[tuple((x, y))] if x < y else -recod[tuple((y, x))]
        self.is_done = True
        if self.tot_siz <= self.minimum_siz:
            score = {x: 0 for x in self.a}
            for i in range(len(self.a)-1):
                for j in range(i+1, len(self.a)):
                    x = self.a[i]
                    y = self.a[j]
                    v = get_recod(x, y)
                    score[x] = score[x] + v
                    score[y] = score[y] - v
            score = [tuple((score[x], x)) for x in self.a]
            score.sort()
            self.ans = [[t[1] for t in score]]
            return False
        else:
            self.ans = [[], []]
            for x in self.a:
                score = 0
                for y in self.bucket:
                    score = score + get_recod(x, y)
                self.ans[0 if(score < 0) else 1].append(x)
            self.need_split = True
            return True

def test():
    n = 100000
    a = [i for i in range(n-1,-1, -1)]
    random.shuffle(a)
    minimum = 200
    mg = quicksort(a, minimum)
    recod = defaultdict()
    cnt = 0
    while (True):
        cnt = cnt + 1
        tmp = []
        for it in mg.seg_list:
            if it.tot_siz > minimum:
                tmp.append(it.tot_siz)
        ask_list = mg.query()
        # print(cnt, ' ', len(recod), ' ', len(mg.seg_list), ' ', set(tmp))
        print(f"cnt:{cnt}  len(recod):{len(recod)} len(ask_list):{len(ask_list)} len(mg.seg_list):{len(mg.seg_list)} {set(tmp)}")
        for it in ask_list:
            if it not in recod:
                recod[it] = 1 if it[0] > it[1] else -1
        if mg.do(recod):
            break

    ans = mg.ans

    assert(len(ans) == n)
    for i in range(n-1):
        if ans[i]>ans[i+1]:
            print("WAWAWA  ", i, ' ', ans[i], ' ',ans[i+1])


if __name__ == "__main__":
    test()
