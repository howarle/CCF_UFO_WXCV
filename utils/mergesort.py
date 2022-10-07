from math import log


class mergesort:
    def __init__(self, a, initial_size=100) -> None:
        self.seg_list = []
        self.ans = None
        self.recod = {}
        self.ask_list = []
        self.first_do = True
        self.is_done = False
        self.merge_list = []

        tot_size = len(a)
        for i in range(0, tot_size, initial_size):
            seg = [a[j] for j in range(i, min(i+initial_size, tot_size))]
            for i in range(len(seg)-1):
                for j in range(i+1, len(seg)):
                    self.__add_ask(seg[i], seg[j])
            self.seg_list.append(seg)

    def __add_ask(self, x, y) -> None:
        if x > y:
            x, y = y, x
        self.ask_list.append(tuple((x, y)))

    def query(self) -> list:
        ask_list = self.ask_list
        self.ask_list = []
        for it in self.merge_list:
            ask_list = ask_list + it.query()

        return ask_list

    def do(self, recod) -> bool:
        assert len(self.ask_list) == 0
        if self.is_done:
            return True

        def get_recod(x, y):
            return recod[tuple((x, y))] if x < y else -recod[tuple((y, x))]

        if self.first_do:
            self.first_do = False
            seg_list = self.seg_list
            self.seg_list = None
            for seg in seg_list:
                score = {x: 0 for x in seg}
                for i in range(len(seg)-1):
                    for j in range(i+1, len(seg)):
                        x = seg[i]
                        y = seg[j]
                        v = get_recod(x, y)
                        score[x] = score[x] + v
                        score[y] = score[y] - v
                score = [tuple((score[x], x)) for x in seg]
                score.sort()
                seg_nw = [t[1] for t in score]

                if self.ans is not None:
                    mg = MergeSegment(self.ans, seg_nw)
                    self.ans = None
                    self.merge_list.append(mg)
                else:
                    self.ans = seg_nw
        else:
            merge_list = []
            for it in self.merge_list:
                if it.do(recod):
                    seg_nw = it.ans
                    if self.ans is not None:
                        mg = MergeSegment(self.ans, seg_nw)
                        self.ans = None
                        merge_list.append(mg)
                    else:
                        self.ans = seg_nw
                else:
                    merge_list.append(it)
            self.merge_list = merge_list

        # for it in 

        if len(self.merge_list) < 1:
            self.is_done = True
        return self.is_done


class MergeSegment:
    def __init__(self, a: list, b: list, bucket_siz=200, k_siz=100) -> None:
        self.tot_siz = len(a) + len(b)
        self.ans = []
        self.ask_list = []

        self.bucket = []
        self.bucket2 = []
        self.score = {}
        self.first_do = True
        self.is_done = False

        self.bucket_siz = bucket_siz + int(log(self.tot_siz, 2))
        self.k_siz = min(int(self.bucket_siz*0.3), k_siz)

        self.belong_a = set(a)
        self.a = a
        self.b = b
        self.a.reverse()
        self.b.reverse()
        self.score = {t: 0 for t in a+b}

        bucket = []

        for i in range(min(self.bucket_siz >> 1, len(a))):
            bucket.append(self.a.pop())

        for i in range(min(self.bucket_siz >> 1, len(b))):
            bucket.append(self.b.pop())

        for i in range(len(bucket)-1):
            for j in range(i+1, len(bucket)):
                self.__add_ask(bucket[i], bucket[j])
        self.bucket = bucket

    def __add_ask(self, x, y) -> None:
        if x > y:
            x, y = y, x
        self.ask_list.append(tuple((x, y)))

    def query(self) -> list:
        tmp = self.ask_list
        self.ask_list = []
        return tmp

    def do(self, recod) -> bool:
        assert len(self.ask_list) == 0
        if self.is_done:
            return True

        def get_recod(x, y):
            return recod[tuple((x, y))] if x < y else -recod[tuple((y, x))]
        if self.first_do:
            self.first_do = False
            for i in range(len(self.bucket)-1):
                for j in range(i+1, len(self.bucket)):
                    x = self.bucket[i]
                    y = self.bucket[j]
                    t = get_recod(x, y)
                    self.score[x] = self.score[x] + t
                    self.score[y] = self.score[y] - t
        else:
            for i in range(len(self.bucket)):
                for j in range(len(self.bucket2)):
                    x = self.bucket[i]
                    y = self.bucket2[j]
                    t = get_recod(x, y)
                    self.score[x] = self.score[x] + t
                    self.score[y] = self.score[y] - t

            for i in range(len(self.bucket2)-1):
                for j in range(i+1, len(self.bucket2)):
                    x = self.bucket2[i]
                    y = self.bucket2[j]
                    t = get_recod(x, y)
                    self.score[x] = self.score[x] + t
                    self.score[y] = self.score[y] - t
            self.bucket = self.bucket + self.bucket2
            self.bucket2 = []

        for _ in range(self.k_siz):
            mi = None
            for t in self.bucket:
                if mi is None or self.score[mi] > self.score[t]:
                    mi = t
                    
            self.bucket.remove(mi)
            self.score[mi] = 0

            # # //////////////////////
            # if len(self.ans)>0:
            #     assert(self.ans[-1] < mi)
            # # //////////////////////

            self.ans.append(mi)
            
            for x in self.bucket:
                t = get_recod(x, mi)
                self.score[x] = self.score[x] - t

            if len(self.a) == 0 and len(self.b) == 0:
                if len(self.bucket2) > 0:
                    break
                tmp = [tuple((self.score[x], x)) for x in self.bucket]
                tmp.sort()
                for t in tmp:
                    self.ans.append(t[1])
                self.is_done = True

                assert(self.tot_siz == len(self.ans))
                return True
            elif (mi in self.belong_a and len(self.a) > 0) or len(self.b) == 0:
                self.bucket2.append(self.a.pop())
            else:
                self.bucket2.append(self.b.pop())

        for i in range(len(self.bucket)):
            for j in range(len(self.bucket2)):
                x = self.bucket[i]
                y = self.bucket2[j]
                self.__add_ask(x, y)

        for i in range(len(self.bucket2)-1):
            for j in range(i+1, len(self.bucket2)):
                x = self.bucket2[i]
                y = self.bucket2[j]
                self.__add_ask(x, y)
        return False

import random
def test():
    n = 90000
    a = [i for i in range(n-1,-1, -1)]
    random.shuffle(a)

    mg = mergesort(a, 500)
    recod = {}
    cnt = 0
    while (True):
        cnt = cnt + 1
        tmp = [it.tot_siz for it in mg.merge_list]
        print(cnt, ' ', len(recod), ' ', len(mg.merge_list), ' ', set(tmp))
        ask_list = mg.query()
        for it in ask_list:
            if it not in recod:
                recod[it] = 1 if it[0] > it[1] else -1
        if mg.do(recod):
            break

    ans = mg.ans

    assert(len(ans) == n)
    for i in range(n-1):
        if ans[i]>ans[i+1]:
            print(i, ' ', ans[i], ' ',ans[i+1])


if __name__ == "__main__":
    test()
