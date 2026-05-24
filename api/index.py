import math
from pathlib import Path
from flask import Flask, request, jsonify, Response

app = Flask(__name__)

HTML_FILE = Path(__file__).parent / "index.html"


def cholesky_decompose(matrix):
    n = len(matrix)
    A = [row[:] for row in matrix]

    for row in A:
        if len(row) != n:
            raise ValueError("Matrix must be square.")

    for i in range(n):
        for j in range(n):
            if abs(A[i][j] - A[j][i]) > 1e-9:
                raise ValueError(
                    "Matrix is not symmetric: "
                    "A[{}][{}]={} != A[{}][{}]={}".format(i, j, A[i][j], j, i, A[j][i])
                )

    L = [[0.0] * n for _ in range(n)]
    steps = []

    for j in range(n):
        sum_sq = sum(L[j][k] ** 2 for k in range(j))
        val = A[j][j] - sum_sq

        if val <= 0:
            raise ValueError(
                "Matrix is not positive-definite. "
                "A[{}][{}] - sum(L[{}][k]^2) = {:.6f} <= 0".format(j, j, j, val)
            )

        L[j][j] = math.sqrt(val)

        formula = "L[{}][{}] = sqrt(A[{}][{}]".format(j+1, j+1, j+1, j+1)
        if j > 0:
            formula += "".join(" - L[{}][{}]^2".format(j+1, k+1) for k in range(j))
        formula += ")"

        steps.append({
            "type": "diagonal",
            "i": j,
            "j": j,
            "formula": formula,
            "sum_sq": round(sum_sq, 8),
            "val": round(val, 8),
            "result": round(L[j][j], 8),
        })

        for i in range(j + 1, n):
            sum_prod = sum(L[i][k] * L[j][k] for k in range(j))
            L[i][j] = (A[i][j] - sum_prod) / L[j][j]

            formula2 = "L[{}][{}] = (A[{}][{}]".format(i+1, j+1, i+1, j+1)
            if j > 0:
                formula2 += "".join(
                    " - L[{}][{}]*L[{}][{}]".format(i+1, k+1, j+1, k+1)
                    for k in range(j)
                )
            formula2 += ") / L[{}][{}]".format(j+1, j+1)

            steps.append({
                "type": "off_diagonal",
                "i": i,
                "j": j,
                "formula": formula2,
                "sum_prod": round(sum_prod, 8),
                "numerator": round(A[i][j] - sum_prod, 8),
                "divisor": round(L[j][j], 8),
                "result": round(L[i][j], 8),
            })

    LT = [[L[j][i] for j in range(n)] for i in range(n)]
    reconstructed = [
        [sum(L[r][k] * LT[k][c] for k in range(n)) for c in range(n)]
        for r in range(n)
    ]

    return {
        "L": [[round(L[i][j], 6) for j in range(n)] for i in range(n)],
        "steps": steps,
        "reconstructed": [[round(reconstructed[i][j], 6) for j in range(n)] for i in range(n)],
        "n": n,
    }


def solve_cholesky(L, b):
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        s = sum(L[i][j] * y[j] for j in range(i))
        y[i] = (b[i] - s) / L[i][i]

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(L[j][i] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - s) / L[i][i]

    return [round(v, 6) for v in x], [round(v, 6) for v in y]


@app.route("/")
def index():
    html = HTML_FILE.read_text(encoding="utf-8")
    return Response(html, mimetype="text/html")


@app.route("/calculate", methods=["POST"])
def calculate():
    try:
        data = request.get_json()
        raw = data.get("matrix", [])
        solve = data.get("solve", False)
        b_vec = data.get("b", [])

        n = len(raw)
        if n < 2 or n > 6:
            return jsonify({"error": "Matrix size must be between 2x2 and 6x6."}), 400

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
        return jsonify({"error": "Unexpected error: {}".format(str(e))}), 500


if __name__ == "__main__":
    app.run(debug=True)
