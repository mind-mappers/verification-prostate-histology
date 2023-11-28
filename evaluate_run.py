from contextlib import redirect_stdout
from homologous_point_prediction.evaluate import evaluate
import sys
import os

logging_dir = sys.argv[1] if len(sys.argv) > 1 else "."
print(logging_dir)
model_name = sys.argv[2]
with open(os.path.join(logging_dir, "evaluate.txt"), "w") as f:
    with redirect_stdout(f):
        evaluate(os.path.join(logging_dir, model_name), logging_dir, requires_scaling=True)