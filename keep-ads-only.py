#!/usr/bin/env python3

import sys
import subprocess

# Fichier source et de sortie
input_file = sys.argv[1]
#output_pattern = "output-%03d.mkv"  # Modifier si un seul fichier voulu
output_pattern = sys.argv[2]

# Plages de timestamps à conserver
ranges = [
    "00:00:00-00:03:51",
    "00:18:42-00:23:46",
    "00:46:50-00:50:05",
    "01:17:16-01:20:26",
    "01:48:50-01:56:57",
    "02:14:20-02:21:38",
    "02:43:09-02:49:28",
    "02:52:29-02:56:51"
]

# Construction de l'argument --split parts:
split_argument = "--split"
#split_value = "parts:" + ",".join(ranges)  # ou ajouter + devant chaque sauf la 1ère si concat en un seul fichier
split_value = "parts:" + ",".join([ranges[0]] + ["+" + r for r in ranges[1:]])
output_pattern = "output.mkv"

cmd = [
    "mkvmerge",
    "-o", output_pattern,
    split_argument, split_value,
    input_file
]

# Affiche la commande pour vérification
print("Commande exécutée :", " ".join(cmd))

# Exécution
try:
    subprocess.run(cmd, check=True)
    print("Découpage terminé avec succès.")
except subprocess.CalledProcessError as e:
    print(f"Erreur lors de l'exécution de mkvmerge : {e}")

