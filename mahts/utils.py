
def format_lstsq_output(output, backend):
    if backend == "lsqr":
        stop_mapping = {1: "x is an approximate solution to Ax = b.",
                        2: "x approximately solves the least-squares problem."}
        print(f"stop: {stop_mapping[output[1]]}")
        print(f"n-iterations: {output[2]}")
        print(f"r1norm norm(r): {output[3]}")
        print(f"r2norm sqrt(norm(r)^2 + damp^2 * norm(x)^2): {output[4]}")
        print(f"A-norm: {output[5]}")
        print(f"A-cond: {output[6]}")
        print(f"Ar-norm: {output[7]}")
        print(f"x-norm: {output[8]}")
    
    if backend == "lsmr":
        stop_mapping = {0: "0. x=0 is a solution.",
                        1: "1. x is an approximate solution to A*x = B, according to atol and btol.",
                        2: "2. x approximately solves the least-squares problem according to atol.",
                        3: "3. COND(A) seems to be greater than CONLIM.",
                        4: "4. is the same as 1 with atol = btol = eps (machine precision).",
                        5: "5. is the same as 2 with atol = eps.",
                        6: "6. is the same as 3 with CONLIM = 1/eps.",
                        7: "7. means ITN reached maxiter before the other stopping conditions were satisfied."}
        print(f"stop: {stop_mapping[output[1]]}")
        print(f"n-iterations: {output[2]}")
        print(f"r1norm norm(r): {output[3]}")
        print(f"Ar-norm: {output[4]}")
        print(f"A-norm: {output[5]}")
        print(f"A-cond: {output[6]}")
        print(f"x-norm: {output[7]}")
