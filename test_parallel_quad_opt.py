import mpmath
import gmpy2
import multiprocessing
import time
import math

def integrand(x):
    return mpmath.exp(mpmath.li(x))

def init_worker(dps):
    mpmath.mp.dps = dps

def worker_chunk(nodes_chunk):
    """
    Takes a chunk of nodes (x, w) in raw tuple format, reconstructs them, 
    evaluates them, and sums the result. Returns the sum as a raw tuple.
    """
    chunk_sum = mpmath.mpf(0)
    for x_raw, w_raw in nodes_chunk:
        x = mpmath.mp.make_mpf((x_raw[0], gmpy2.mpz(x_raw[1]), x_raw[2], x_raw[3]))
        w = mpmath.mp.make_mpf((w_raw[0], gmpy2.mpz(w_raw[1]), w_raw[2], w_raw[3]))
        chunk_sum += w * integrand(x)
        
    sign, man, exp, bc = chunk_sum._mpf_
    return (int(sign), int(man), int(exp), int(bc))

class ParallelTanhSinh(mpmath.calculus.quadrature.TanhSinh):
    def __init__(self, ctx, pool, num_cores=12):
        super().__init__(ctx)
        self.pool = pool
        self.num_cores = num_cores

    def sum_next(self, f, nodes, degree, prec, previous, verbose=False):
        print(f"Integrating degree {degree} with {len(nodes)} nodes...")
        
        # Format nodes for low-overhead IPC
        raw_nodes = []
        for x, w in nodes:
            xs, xm, xe, xbc = x._mpf_
            ws, wm, we, wbc = w._mpf_
            x_raw = (int(xs), int(xm), int(xe), int(xbc))
            w_raw = (int(ws), int(wm), int(we), int(wbc))
            raw_nodes.append((x_raw, w_raw))
            
        # Split into exactly num_cores chunks
        chunk_size = math.ceil(len(raw_nodes) / self.num_cores)
        if chunk_size == 0:
            chunk_size = 1
            
        chunks = [raw_nodes[i:i + chunk_size] for i in range(0, len(raw_nodes), chunk_size)]
        
        total = self.ctx.mpf(0)
        # Process chunks in parallel
        results = self.pool.map(worker_chunk, chunks)
        
        for res_raw in results:
            sign, man, exp, bc = res_raw
            chunk_val = mpmath.mp.make_mpf((sign, gmpy2.mpz(man), exp, bc))
            total += chunk_val
            
        return total

if __name__ == '__main__':
    dps = 1000
    mpmath.mp.dps = dps
    
    t0 = time.time()
    
    num_cores = 12
    with multiprocessing.Pool(processes=num_cores, initializer=init_worker, initargs=(dps,)) as pool:
        rule = ParallelTanhSinh(mpmath.mp, pool, num_cores)
        mpmath.mp.TanhSinh = lambda ctx: rule
        
        result = mpmath.quad(integrand, [0, 1], method='tanh-sinh')
        
    t1 = time.time()
    print("Optimized parallel time:", t1 - t0)
    print("Result:", result)
