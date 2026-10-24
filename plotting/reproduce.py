"""Regenerate only the figures included in the current QGSC manuscript."""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PYTHON_FIGURES = {
    1: 'generate_fig01_from_drawio.py', 3: 'plot_fig03_coverage.py',
    5: 'plot_fig05_decomposition.py', 7: 'reflow_fig07_cxy.py',
    8: 'fig08_bodyfont.py', 10: 'plot_fig10_clear.py',
    12: 'plot_restricted_dual_axis.py',
}
DRAWIO_FIGURES = {
    2: [('fig02_weyl_revised.drawio', 'fig02_weyl_revised_v3.pdf')],
    4: [('fig04_ac2_revised.drawio', 'fig04_ac2_revised_v3.pdf')],
    9: [(f'fig10_panel_{p}_v2.drawio', f'fig10_panel_{p}_v2.pdf') for p in 'abc'],
}


def run(command):
    subprocess.run(command, cwd=ROOT, check=True, timeout=60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure', nargs='+', type=int, choices=range(1, 13))
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--drawio', default=os.environ.get('DRAWIO_BIN') or shutil.which('draw.io'))
    parser.add_argument('--xelatex', default=os.environ.get('XELATEX_BIN') or shutil.which('xelatex'))
    args = parser.parse_args()
    numbers = list(range(1, 13)) if args.all else args.figure
    if not numbers:
        parser.error('Use --figure NUMBER [NUMBER ...] or --all.')
    # Check external tools before overwriting any existing images.
    if set(numbers) & set(DRAWIO_FIGURES) and not args.drawio:
        parser.error('Figures 2, 4 and 9 require --drawio PATH or DRAWIO_BIN.')
    if set(numbers) & {6, 11} and not args.xelatex:
        parser.error('Figures 6 and 11 require --xelatex PATH or XELATEX_BIN.')
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT.parent/'outputs'/'matplotlib-config'))
    for number in numbers:
        print(f'Figure {number}', flush=True)
        if number in PYTHON_FIGURES:
            run([sys.executable, str(ROOT/'plotting'/PYTHON_FIGURES[number])])
        elif number in DRAWIO_FIGURES:
            for source, target in DRAWIO_FIGURES[number]:
                run([args.drawio, '--export', '--format', 'pdf', '--crop',
                     '--output', str(ROOT/'assets'/target), str(ROOT/'figures'/source)])
        else:
            run([args.xelatex, '-interaction=nonstopmode', '-halt-on-error',
                 '-output-directory='+str(ROOT/'assets'),
                 str(ROOT/'plotting'/f'figure_{number:02d}_standalone.tex')])


if __name__ == '__main__':
    main()
