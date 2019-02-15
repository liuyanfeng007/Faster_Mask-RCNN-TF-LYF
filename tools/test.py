
def integers_starting_from(i):
    while True:
        yield i
        i += 1

stream = integers_starting_from(3)
print(stream.__next__())  # 打印3
print(stream.__next__())  # 打印4

def stream_map(func, stream):
    while True:
        yield func(stream.next())

def stream_filter(pred, stream):
    while True:
        x = stream.__next__()
        if pred(x):
            yield x

def sieve():
    def divisible(x):
        return lambda e: e % x != 0
    stream = integers_starting_from(2)
    while True:
        x = stream.__next__()
        yield x
        stream = stream_filter(divisible(x),stream)

def printn(n, stream):
    for _ in range(n):
        print(stream.__next__()),
    print("----------------------------------")

printn(100, sieve())

pass