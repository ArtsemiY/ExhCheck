"""ExhCheck is software aimed to check the minimal value 
of exhaustiveness parameter for the AutoDock Vina, which 
enables to reproduce docking results in a stable manner.

This is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ExhCheck is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Copyright 2020 Artsemi M. Yushkevich, Alexander M. Andrianov. Drug Design Group, The United Institute of Informatics Problem.
If you have any questions, comments, or suggestions, please don't hesitate to contact me at fpm.yushkeviAM [at] bsu [dot] by.
"""

from pathlib import Path
from tqdm import tqdm
from scipy.interpolate import CubicSpline
import subprocess as sp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections

#set folders to use 
basedir = Path(__file__).resolve().parent
input_dir = Path (basedir) / 'input'
ligand_filename = Path(input_dir) / 'ligand.pdbqt'
receptor_filename = Path(input_dir) / 'receptor.pdbqt'
vina_out_dir = Path (basedir) / 'vina_out' 
logs_dir = Path(basedir) / 'logs'

# set grid box center coordinates 
center_x = 29.0
center_y = -14.6
center_z = 37.7

# set grid box sizes
size_x = 17.0
size_y = 16.4
size_z = 21.1

#set exhaustiveness (exh) boundaries and step
exh_min = 5
exh_max = 45
exh_step = 10

num_of_runs = 3

#ligands = [ligands_dir.joinpath(f) for f in ligands_dir.iterdir() if ligands_dir.joinpath(f).is_file()]
ligand = ligand_filename.stem

#set default for current ligand and receptor docking parameters
to_execute_default = (
	'vina'
	' --receptor {receptor}'
	' --ligand {ligand}'
	' --center_x {center_x}'
	' --center_y {center_y}'
	' --center_z {center_z}'
	' --size_x {size_x}'
	' --size_y {size_y}'
	' --size_z {size_z}'.format(
		receptor=receptor_filename,
		ligand=ligand_filename, 
		center_x=str(center_x),
		center_y=str(center_y),
		center_z=str(center_z),
		size_x=str(size_x),
		size_y=str(size_y),
		size_z=str(size_z))
	)

print('Ligand: {ligand}'.format(ligand=ligand))


#alteration of exhaustiveness
for exh in tqdm(range(exh_min, exh_max+exh_step, exh_step), desc='EXH'):
	print('Exhaustiveness: {exh}'.format(exh=exh))

	log_filename = Path(logs_dir) / ('log_{ligand}_exh{exh}.txt'.format(ligand=ligand, exh=exh))	
	
	#make several runs to accumulate variability of results
	for run in tqdm(range(1, num_of_runs+1), desc='RUN', leave=False):

		out_filename = Path(vina_out_dir) / ('{ligand}_out_exh_{exh}_run_{run}.pdbqt'.format(ligand=ligand, exh=exh, run=run))
		to_execute = to_execute_default + ' --out {out} --exhaustiveness {exh}'.format(out=out_filename, exh=str(exh))		
		cmd = list(to_execute.split(" "))

		#get Vina output
		vina_out = sp.check_output(cmd)

		#append VINA log to log file of current run
		with open(log_filename, 'a') as file:
			print('='*10)
			print('RUN #{run}'.format(run=run), file=file)
			print(vina_out.decode("utf-8"), file=file)


#get scores
scores = collections.defaultdict(list)
vina_outputs = [vina_out_dir.joinpath(f) for f in vina_out_dir.iterdir() if vina_out_dir.joinpath(f).is_file()]
for run_vina_out in vina_outputs:

	run_name = run_vina_out.stem
	i_exh = run_name.find('exh_')
	run_name = run_name[i_exh:]
	gap1, exh, gap2, run = run_name.split('_') 

	with open(run_vina_out, 'r') as f:
		score = f.readline()
		while 'VINA RESULT' not in score:
			score = f.readline()
		score = score.split()
		score = float(score[3])
		scores[exh].append(score)
	scores[exh] = sorted(scores[exh])

results = pd.DataFrame.from_dict(scores)
results.sort_index(axis=1, ascending=True, inplace=True)
results = results.stack()
results = results.reset_index(level=0, drop=True).reset_index(name='Energy').rename(columns={'index':'Exhaustiveness'})
results['Exhaustiveness'] = results['Exhaustiveness'].astype('int')
worst_energies = results.groupby(['Exhaustiveness']).max(level='Energy')
best_energies = results.groupby(['Exhaustiveness']).min(level='Energy')

n_exh = ((exh_max-exh_min)/exh_step) * 100
exh_smooth = np.linspace(exh_min, exh_max, n_exh)

i_worst = np.interp(x=exh_smooth, xp=worst_energies.index.tolist(), fp=worst_energies['Energy'].tolist())
i_best = np.interp(x=exh_smooth, xp=best_energies.index.tolist(), fp=best_energies['Energy'].tolist())


#i_worst = CubicSpline(x=worst_energies.index.tolist(), y=worst_energies['Energy'].tolist())
#i_best = CubicSpline(x=best_energies.index.tolist(), y=best_energies['Energy'].tolist())

energies_to_result = pd.DataFrame()
energies_to_result['Worst'] = i_worst #(exh_smooth)
energies_to_result['Best'] = i_best #(exh_smooth)
energies_to_result.index = exh_smooth

energies_to_result['Range'] = energies_to_result['Best'] - energies_to_result['Worst']

print(energies_to_result)

plt.figure()

plt.plot(exh_smooth, energies_to_result['Worst'].tolist(), 'r-')
plt.plot(exh_smooth, energies_to_result['Best'].tolist(), 'g-')

plt.scatter(x=worst_energies.index.tolist(), y=worst_energies['Energy'].tolist(), color='r', label='worst')
plt.scatter(x=best_energies.index.tolist(), y=best_energies['Energy'].tolist(), color='g', label = 'best')

plt.xlabel('Exhaustiveness')
plt.ylabel('Energy, kcal/mol')

plt.minorticks_on()
plt.grid(which='major', linestyle='-', color='black')
plt.grid(which='minor', linestyle=':', color='gray')

plt.title('Exhaustiveness range change, ligand: {ligand}'.format(ligand=ligand))
plt.legend()

ybottom, ytop = plt.ylim()
plt.ylim(ytop, ybottom)

plt.show()
plt.savefig(Path(basedir) / 'results.pdf')

print('Program terminated')