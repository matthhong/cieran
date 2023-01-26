



class Foo:

    x = 1
    bar = 2
    me = 4

    def __init__(self):
        pass

    def __str__(self):
        return "{}, {}, {}".format(self.x, self.bar, self.me)



if __name__=='__main__':
    print(Foo())