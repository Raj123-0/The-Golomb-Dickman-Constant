import os
import sys
import gc
import math
import argparse
import multiprocessing

# Ensure gmpy2 is used by mpmath for core arithmetic performance.
import gmpy2
import mpmath

try:
    from tqdm import tqdm
except ImportError:
    print("Please install tqdm: pip install tqdm")
    sys.exit(1)

def integrand(x):
    """
    The mathematically optimal integrand for the Golomb-Dickman constant:
    lambda = int_0^1 exp(li(x)) dx
    """
    return mpmath.exp(mpmath.li(x))

def init_worker(dps):
    """
    Initializes the worker process precision state.
    """
    mpmath.mp.dps = dps

def worker_chunk(nodes_chunk):
    """
    Takes a chunk of nodes (x, w) in raw tuple format, reconstructs them, 
    evaluates them, and sums the result. Returns the sum as a raw tuple.
    This entirely avoids the heavy IPC overhead of pickling large mpf objects.
    """
    chunk_sum = mpmath.mpf(0)
    for x_raw, w_raw in nodes_chunk:
        x = mpmath.mp.make_mpf((x_raw[0], gmpy2.mpz(x_raw[1]), x_raw[2], x_raw[3]))
        w = mpmath.mp.make_mpf((w_raw[0], gmpy2.mpz(w_raw[1]), w_raw[2], w_raw[3]))
        chunk_sum += w * integrand(x)
        
    sign, man, exp, bc = chunk_sum._mpf_
    return (int(sign), int(man), int(exp), int(bc))

def node_worker_chunk(args):
    """
    Computes a chunk of quadrature nodes in parallel to bypass the 
    single-threaded node generation bottleneck.
    """
    start_k, end_k, degree, prec, dps = args
    mpmath.mp.dps = dps
    ctx = mpmath.mp
    extra = 20
    ctx.prec += extra
    tol = ctx.ldexp(1, -prec-10)
    pi4 = ctx.pi/4
    
    t0 = ctx.ldexp(1, -degree)
    h = t0 if degree == 1 else t0*2
    
    # Calculate initial a and b for start_k to jumpstart the iterative loop
    t_start = t0 + start_k * h
    expt = ctx.exp(t_start)
    a = pi4 * expt
    b = pi4 / expt
    
    udelta = ctx.exp(h)
    urdelta = 1/udelta
    
    nodes_raw = []
    
    for k in range(start_k, end_k):
        c = ctx.exp(a-b)
        d = 1/c
        co = (c+d)/2
        si = (c-d)/2
        x = si / co
        w = (a+b) / co**2
        
        diff = abs(x-1)
        if diff <= tol:
            break
            
        xs, xm, xe, xbc = x._mpf_
        ws, wm, we, wbc = w._mpf_
        xr = (int(xs), int(xm), int(xe), int(xbc))
        wr = (int(ws), int(wm), int(we), int(wbc))
        
        nodes_raw.append((xr, wr))
        # Add the negative node counterpart
        nodes_raw.append(((int(not xs), int(xm), int(xe), int(xbc)), wr))
        
        a *= udelta
        b *= urdelta
        
    return nodes_raw

class ParallelTanhSinh(mpmath.calculus.quadrature.TanhSinh):
    """
    A custom Tanh-Sinh quadrature rule that hijacks both the sequential node generation 
    and summation steps, distributing them perfectly across multiple CPU cores.
    """
    def __init__(self, ctx, pool, num_cores):
        super().__init__(ctx)
        self.pool = pool
        self.num_cores = num_cores

    def calc_nodes(self, degree, prec, verbose=False):
        # Fall back to sequential for very low degrees to avoid IPC overhead
        if degree < 5:
            return super().calc_nodes(degree, prec, verbose)
            
        ctx = self.ctx
        
        # Estimate max_k using safe mathematical boundaries
        t0_float = 2.0**(-degree)
        h_float = t0_float if degree == 1 else t0_float * 2
        
        target_sinh = (prec + 10) * math.log(2) / math.pi
        target_t = math.asinh(target_sinh)
        max_k = int(math.ceil((target_t - t0_float) / h_float)) + 100 # Add safety buffer
        
        print(f"\n[Degree {degree}] Generating ~{max_k*2} quadrature nodes across {self.num_cores} cores...")
        
        chunk_size = math.ceil(max_k / (self.num_cores * 4))
        if chunk_size == 0: chunk_size = 1
        
        chunks = []
        for i in range(0, max_k, chunk_size):
            chunks.append((i, min(i+chunk_size, max_k), degree, prec, ctx.dps))
            
        nodes = []
        if degree == 1:
            nodes.append((ctx.zero, ctx.pi/2))
            
        # Distribute node calculation
        results = self.pool.imap(node_worker_chunk, chunks)
        
        for res_raw in tqdm(results, total=len(chunks), desc=f"Node Gen Degree {degree}", unit="chunk"):
            for xr, wr in res_raw:
                x = ctx.make_mpf((xr[0], gmpy2.mpz(xr[1]), xr[2], xr[3]))
                w = ctx.make_mpf((wr[0], gmpy2.mpz(wr[1]), wr[2], wr[3]))
                nodes.append((x, w))
                
        return nodes

    def sum_next(self, f, nodes, degree, prec, previous, verbose=False):
        # If the number of nodes is small (e.g. initial low degrees),
        # compute them sequentially to avoid IPC overhead.
        if len(nodes) < 50:
            return super().sum_next(f, nodes, degree, prec, previous, verbose)

        # We are at a high degree, distribute the work!
        print(f"\n[Degree {degree}] Distributing {len(nodes)} quadrature evaluations across {self.num_cores} cores...")
        
        # Format nodes for low-overhead IPC
        raw_nodes = []
        for x, w in nodes:
            xs, xm, xe, xbc = x._mpf_
            ws, wm, we, wbc = w._mpf_
            x_raw = (int(xs), int(xm), int(xe), int(xbc))
            w_raw = (int(ws), int(wm), int(we), int(wbc))
            raw_nodes.append((x_raw, w_raw))
            
        # Split into chunks (we multiply num_cores by 4 to get better load balancing 
        # and smoother tqdm updates, without incurring too much IPC overhead)
        chunk_size = math.ceil(len(raw_nodes) / (self.num_cores * 4))
        if chunk_size == 0:
            chunk_size = 1
            
        chunks = [raw_nodes[i:i + chunk_size] for i in range(0, len(raw_nodes), chunk_size)]
        
        total = self.ctx.mpf(0)
        
        # Process chunks in parallel with a tqdm progress bar & ETA
        for res_raw in tqdm(self.pool.imap_unordered(worker_chunk, chunks), total=len(chunks), desc=f"Integration Degree {degree}", unit="chunk"):
            sign, man, exp, bc = res_raw
            chunk_val = mpmath.mp.make_mpf((sign, gmpy2.mpz(man), exp, bc))
            total += chunk_val
            
        return total

def save_oeis_bfile(digits_str, filename):
    """
    Formats and saves the digits string into a standard OEIS b-file format.
    Format: [index] [digit] separated by a space on each line.
    """
    with open(filename, "w") as f:
        # OEIS sequences generally start at index 1 for the first term
        # The Golomb-Dickman constant is approx 0.6243299...
        # Its decimal expansion sequence (A084945) begins with the first decimal digit: 6
        for index, digit in enumerate(digits_str, start=1):
            f.write(f"{index} {digit}\n")

def calculate_golomb_dickman(n_digits):
    """
    Calculates the Golomb-Dickman constant to exactly n_digits using parallel integration,
    and saves the outputs according to strict formatting rules.
    """
    padding = 50
    working_dps = n_digits + padding
    mpmath.mp.dps = working_dps
    
    # Cap CPU utilisation to 75% to keep the system responsive as requested
    total_cores = multiprocessing.cpu_count()
    num_cores = max(1, int(math.ceil(total_cores * 0.75)))
        
    print(f"Target digits: {n_digits}")
    print(f"Working precision (dps): {working_dps}")
    print(f"Starting optimized multiprocessing pool with {num_cores} CPU cores (75% utilisation)...")
    
    # We use a context manager for the pool to ensure resources are cleaned up
    with multiprocessing.Pool(processes=num_cores, initializer=init_worker, initargs=(working_dps,)) as pool:
        # Inject the parallel TanhSinh rule into mpmath
        rule = ParallelTanhSinh(mpmath.mp, pool, num_cores)
        mpmath.mp.TanhSinh = lambda ctx: rule
        
        # Compute the integral over [0, 1] using our monkey-patched parallel quadrature algorithm
        t0=time.time()
total_val = mpmath.quad(integrand, [0, 1], method='tanh-sinh')
print('Quad time:', time.time() - t0)
            
    print("\nIntegration complete. Applying strict truncation...")
            
    # TRUNCATION (NO ROUNDING)
    multiplier = mpmath.power(10, n_digits)
    shifted = total_val * multiplier
    
    # Apply math.floor to strictly drop all remaining fractional digits.
    int_part = mpmath.floor(shifted)
    
    # Convert exactly to integer, then to string. gmpy2 handles massive integers instantly.
    digits_str = str(int(int_part))
    
    # Ensure zero-padding in the extremely unlikely event leading zeros exist
    if len(digits_str) < n_digits:
        digits_str = digits_str.zfill(n_digits)
        
    # Final sanity check to ensure strict output length matches exactly n_digits
    digits_str = digits_str[:n_digits]
    
    # Save the standard text file
    output_filename = f"The Golomb-Dickman Constant_{n_digits}_digits.txt"
    with open(output_filename, "w") as f:
        # A single continuous string of text without any decimal points, spaces, or hard line breaks
        f.write(digits_str)
    print(f"Saved strictly truncated digits to: {output_filename}")
        
    # Save the OEIS b-file format
    bfile_filename = f"b084945_{n_digits}_digits.txt"
    save_oeis_bfile(digits_str, bfile_filename)
    print(f"Saved OEIS b-file to: {bfile_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the Golomb-Dickman Constant to N significant digits.")
    parser.add_argument("digits", type=int, nargs="?", default=1000, help="Number of digits to calculate.")
    args = parser.parse_args()
    
    calculate_golomb_dickman(args.digits)
