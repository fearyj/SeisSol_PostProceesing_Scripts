import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
        '--input_file',
        default='nuc3output.csv',
        )
args = parser.parse_args()

input_file = Path(args.input_file)
fname = input_file.name[:-4]

print(input_file)

df = pd.read_csv(input_file)
#xmin, xmax = df['lon'].min(), df['lon'].max()
#ymin, ymax = df['lat'].min(), df['lat'].max()

#take slip != 0.0 m only
df = df[df['slip'] !=0.0]

# plot
fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
ax.coastlines(zorder=0)

slip = ax.scatter(df['lon'], df['lat'], c=df['slip'], zorder=1,
                  s=1)

fig.colorbar(slip, orientation='vertical', label='slip (m)')

ax.set_xlim(df['lon'].min()-1, df['lon'].max()+1)
ax.set_ylim(df['lat'].min()-1, df['lat'].max()+1)
fig.savefig(f'output_ffm_slip__{fname}.png', dpi=300)
plt.close()
