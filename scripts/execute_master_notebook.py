"""
Master Notebook Automated Runner & Output Serializer.
Executes all cells (Sel 1 to Sel 9) in notebooks/03_production_slu_deep_learning_pipeline.ipynb,
capturing real-time logs, stdout, and matplotlib spectrogram/curve figures directly into the .ipynb file.
"""

import sys
import os
import io
import time
import base64
import traceback
from pathlib import Path
import nbformat
from nbformat.v4 import new_output

# Set environment for headless Matplotlib and offline HuggingFace
os.environ['MPLBACKEND'] = 'Agg'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

NOTEBOOK_PATH = Path('notebooks/03_production_slu_deep_learning_pipeline.ipynb').resolve()

class DualWriter:
    """Writes to both the real terminal/stdout and an in-memory buffer."""
    def __init__(self, original_stream):
        self.original_stream = original_stream
        self.buffer = io.StringIO()

    def write(self, s):
        self.original_stream.write(s)
        self.original_stream.flush()
        self.buffer.write(s)

    def flush(self):
        self.original_stream.flush()
        self.buffer.flush()

    def get_value(self):
        return self.buffer.getvalue()

    def clear(self):
        self.buffer = io.StringIO()

def run_notebook():
    print(f"[RUNNER] Loading notebook: {NOTEBOOK_PATH}")
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    print(f"[RUNNER] Total cells found: {len(nb.cells)}")

    # Shared execution namespace across all cells
    execution_namespace = {
        '__name__': '__main__',
        '__file__': str(NOTEBOOK_PATH),
    }

    execution_count = 1
    total_code_cells = sum(1 for c in nb.cells if c.cell_type == 'code')
    current_code_idx = 0

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for cell_idx, cell in enumerate(nb.cells):
        if cell.cell_type != 'code':
            continue

        current_code_idx += 1
        source_lines = cell.source.splitlines()
        header = source_lines[0] if source_lines else 'Code Cell'
        for line in source_lines:
            if 'SEL ' in line:
                header = line.strip('# =').strip()
                break

        print(f"\n{'='*75}")
        print(f"[RUNNER] ({current_code_idx}/{total_code_cells}) EXECUTING: {header} (Cell Index {cell_idx})")
        print(f"{'='*75}\n")

        # Captured outputs for this cell
        cell_outputs = []
        captured_figures = []

        # Hook plt.show() to capture plots as display_data
        original_plt_show = plt.show

        def custom_plt_show(*args, **kwargs):
            try:
                fig = plt.gcf()
                if fig is not None:
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
                    buf.seek(0)
                    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                    captured_figures.append(img_base64)
                    plt.close(fig)
            except Exception as e:
                print(f"[WARN] Failed to capture figure: {e}")

        plt.show = custom_plt_show

        # Dual writer for stdout and stderr
        stdout_interceptor = DualWriter(sys.__stdout__)
        stderr_interceptor = DualWriter(sys.__stderr__)

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = stdout_interceptor
        sys.stderr = stderr_interceptor

        start_time = time.time()
        has_error = False
        error_info = None

        try:
            exec(cell.source, execution_namespace)
        except Exception as e:
            has_error = True
            exc_type, exc_val, exc_tb = sys.exc_info()
            tb_lines = traceback.format_exception(exc_type, exc_val, exc_tb)
            error_info = (exc_type.__name__, str(exc_val), tb_lines)
            print(f"\n[ERROR] Exception occurred in cell {cell_idx}:\n{''.join(tb_lines)}", file=sys.stderr)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            plt.show = original_plt_show

        elapsed = time.time() - start_time
        print(f"\n[RUNNER] Cell {cell_idx} completed in {elapsed:.2f}s (Error: {has_error})")

        # Compile captured outputs
        stdout_text = stdout_interceptor.get_value()
        stderr_text = stderr_interceptor.get_value()

        if stdout_text.strip():
            cell_outputs.append(new_output(
                output_type='stream',
                name='stdout',
                text=stdout_text
            ))

        if stderr_text.strip():
            cell_outputs.append(new_output(
                output_type='stream',
                name='stderr',
                text=stderr_text
            ))

        for fig_b64 in captured_figures:
            cell_outputs.append(new_output(
                output_type='display_data',
                data={
                    'image/png': fig_b64,
                    'text/plain': '<Figure size ...>'
                },
                metadata={}
            ))

        if has_error and error_info:
            ename, evalue, tb_list = error_info
            cell_outputs.append(new_output(
                output_type='error',
                ename=ename,
                evalue=evalue,
                traceback=tb_list
            ))

        # Assign outputs and execution count
        cell.outputs = cell_outputs
        cell.execution_count = execution_count
        execution_count += 1

        # Save notebook checkpoint after each cell
        with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f"[RUNNER] Notebook saved to disk with updated Cell {cell_idx} outputs.")

        if has_error:
            print(f"[RUNNER] Halting execution due to error in cell {cell_idx}.")
            sys.exit(1)

    print(f"\n{'='*75}")
    print("[RUNNER] ALL CODE CELLS (SEL 1 - SEL 9) EXECUTED SUCCESSFULLY!")
    print(f"[RUNNER] Master notebook updated at: {NOTEBOOK_PATH}")
    print(f"{'='*75}\n")

if __name__ == '__main__':
    run_notebook()
