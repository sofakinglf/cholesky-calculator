import os
import math
import json
from flask import Flask, render_template, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, '..', 'templates')
)
def cholesky_decompose(matrix):
    """
    Perform Cholesky decomposition on a symmetric positive-definite matrix.
    Returns L such that A = L @ L^T, along with step-by-step details.

    Args:
        matrix: list of lists (n x n symmetric positive-definite matrix)

    Returns:
        dict with L matrix, steps, and verification info
    """
    n = len(matrix)
    A = [row[:] for row in matrix]  # deep copy

    # Validate square
    for row in A:
        if len(row) != n:
            raise ValueError("Matrix must be square.")

    # Validate symmetry
    for i in range(n):
        for j in range(n):
            if abs(A[i][j] - A[j][i]) > 1e-9:
                raise ValueError(
                    f"Matrix is not symmetric: A[{i}][{j}]={A[i][j]} ≠ A[{j}][{i}]={A[j][i]}"
                )

    # Initialize L
    L = [[0.0] * n for _ in range(n)]
    steps = []

    for j in range(n):
        # Diagonal element
        sum_sq = sum(L[j][k] ** 2 for k in range(j))
        val = A[j][j] - sum_sq

        if val <= 0:
            raise ValueError(
                f"Matrix is not positive-definite. "
                f"A[{j}][{j}] - Σ L[{j}][k]² = {val:.6f} ≤ 0"
            )

        L[j][j] = math.sqrt(val)

        step = {
            "type": "diagonal",
            "i": j,
            "j": j,
            "formula": f"L[{j+1}][{j+1}] = sqrt(A[{j+1}][{j+1}]"
                       + ("".join(f" - L[{j+1}][{k+1}]²" for k in range(j)) if j > 0 else "")
                       + ")",
            "sum_sq": round(sum_sq, 8),
            "val": round(val, 8),
            "result": round(L[j][j], 8),
        }
        steps.append(step)

        # Sub-diagonal elements
        for i in range(j + 1, n):
            sum_prod = sum(L[i][k] * L[j][k] for k in range(j))
            L[i][j] = (A[i][j] - sum_prod) / L[j][j]

            step = {
                "type": "off_diagonal",
                "i": i,
                "j": j,
                "formula": f"L[{i+1}][{j+1}] = (A[{i+1}][{j+1}]"
                           + ("".join(f" - L[{i+1}][{k+1}]*L[{j+1}][{k+1}]" for k in range(j)) if j > 0 else "")
                           + f") / L[{j+1}][{j+1}]",
                "sum_prod": round(sum_prod, 8),
                "numerator": round(A[i][j] - sum_prod, 8),
                "divisor": round(L[j][j], 8),
                "result": round(L[i][j], 8),
            }
            steps.append(step)

    # Verification: compute L @ L^T
    LT = [[L[j][i] for j in range(n)] for i in range(n)]
    reconstructed = [[sum(L[r][k] * LT[k][c] for k in range(n)) for c in range(n)] for r in range(n)]

    # Round results
    L_rounded = [[round(L[i][j], 6) for j in range(n)] for i in range(n)]
    recon_rounded = [[round(reconstructed[i][j], 6) for j in range(n)] for i in range(n)]

    return {
        "L": L_rounded,
        "steps": steps,
        "reconstructed": recon_rounded,
        "n": n,
    }


def solve_cholesky(L, b):
    """
    Solve Ax = b given L (Cholesky factor) using forward/backward substitution.
    """
    n = len(L)
    # Forward substitution: Ly = b
    y = [0.0] * n
    for i in range(n):
        s = sum(L[i][j] * y[j] for j in range(i))
        y[i] = (b[i] - s) / L[i][i]

    # Backward substitution: L^T x = y
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(L[j][i] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - s) / L[i][i]

    return [round(v, 6) for v in x], [round(v, 6) for v in y]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/calculate", methods=["POST"])
def calculate():
    try:
        data = request.get_json()
        raw = data.get("matrix", [])
        solve = data.get("solve", False)
        b_vec = data.get("b", [])

        # Parse matrix
        n = len(raw)
        if n < 2 or n > 6:
            return jsonify({"error": "Matrix size must be between 2×2 and 6×6."}), 400

        matrix = []
        for row in raw:
            if len(row) != n:
                return jsonify({"error": "Matrix must be square."}), 400
            matrix.append([float(x) for x in row])

        result = cholesky_decompose(matrix)

        if solve and b_vec:
            b = [float(v) for v in b_vec]
            if len(b) != n:
                return jsonify({"error": "Vector b must have same length as matrix."}), 400
            x, y = solve_cholesky(result["L"], b)
            result["x"] = x
            result["y"] = y
            result["b"] = [round(v, 6) for v in b]

        return jsonify(result)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
