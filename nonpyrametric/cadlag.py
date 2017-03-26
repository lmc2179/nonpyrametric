class CadlagStepFunction(object):
    def __init__(self, X, Y, invertible=True):
        self.lookup = {x: y for x, y in zip(X, Y)}
        self.X, self.Y = zip(*sorted(zip(X, Y)))
        if invertible:
            self.inverse = CadlagStepFunction(Y, X, invertible=False)
        else:
            self.inverse = None

    def get_X(self):
        return self.X

    def get_Y(self):
        return self.Y

    def __call__(self, x):
        if x in self.lookup:
            return self.lookup[x]
        else:
            x_nearest = self._find_nearest_bound(x, self.X, 0, len(self.X)-1)
            return self.lookup[x_nearest]

    def _find_nearest_bound(self, x, A, l, r):
        i = int((l+r)/2)
        # print(l, i, r, x, A[i])
        if x == A[i]:
            return A[i]
        if x > A[i]:
            if i == len(A)-1:
                return A[-1]
            else:
                if x < A[i+1]:
                    return A[i]
                else:
                    return self._find_nearest_bound(x, A, i+1, r)
        elif x < A[i]:
            if i == 0:
                return A[0]
            else:
                return self._find_nearest_bound(x, A, l, i)