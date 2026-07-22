import mpmath
import multiprocessing
import time

def integrand(x):
    return mpmath.exp(mpmath.li(x))

def init_worker(dps):
    mpmath.mp.dps = dps

def worker(args):
    x, w = args
    return (w, integrand(x))

class ParallelTanhSinh(mpmath.calculus.quadrature.TanhSinh):
    def __init__(self, ctx, pool, chunksize=10):
        super().__init__(ctx)
        self.pool = pool
        self.chunksize = chunksize

    def sum_next(self, f, nodes, degree, prec, previous, verbose=False):
        print(f"Integrating degree {degree} with {len(nodes)} nodes...")
        # nodes is a list of (x, w)
        args_list = [(x, w) for x, w in nodes]
        total = self.ctx.mpf(0)
        # Using map to get results and sum them
        results = self.pool.imap_unordered(worker, args_list, chunksize=self.chunksize)
        for w, fx in results:
            total += w * fx
        return total

if __name__ == '__main__':
    dps = 1000
    mpmath.mp.dps = dps
    
    t0 = time.time()
    
    with multiprocessing.Pool(processes=12, initializer=init_worker, initargs=(dps,)) as pool:
        rule = ParallelTanhSinh(mpmath.mp, pool)
        # We need to tell mpmath to use our rule.
        # But mpmath.quad doesn't accept a rule instance directly in an easy way.
        # quad method='tanh-sinh' looks up the class in ctx. 
        # But we can just call rule.summation directly.
        # Or we can patch mpmath.calculus.quadrature.TanhSinh temporarily!
        
        # Actually let's just do:
        mpmath.mp.TanhSinh = lambda ctx: rule
        
        result = mpmath.quad(integrand, [0, 1], method='tanh-sinh')
        
    t1 = time.time()
    print("Parallel time:", t1 - t0)
    print("Result:", result)
