#!/usr/bin/env python3

import cv2
import sys

video_path = sys.argv[1]
folder_output = sys.argv[2]

cap = cv2.VideoCapture(video_path)
count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Extraction de la ROI du logo
    # LOGO M6 en haut à droite
    ##coord_x = 1740
    ##coord_y = 65
    ##largeur = 85
    ##hauteur = 95
    # LOGO CPVA en bas à gauche
    coord_x = 120
    coord_y = 905
    largeur = 163
    hauteur = 103
    x, y, w, h = roi = (coord_x, coord_y, largeur, hauteur)  # à définir
    logo_region = frame[y:y+h, x:x+w]
    
    # Sauvegarder l'image de la ROI
    cv2.imwrite(f"{folder_output}/frame_{count}.png", logo_region)

    sys.stdout.write(f'\r{count}')  # Réécrit la ligne avec la valeur de count
    sys.stdout.flush()  # Force l'affichage immédiat
    
    count += 1

cap.release()

