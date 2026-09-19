#!/bin/bash
# Reset 00:00 UTC — fila de 5 submits (veredito subagente 3)
K=.venv/bin/kaggle
C=previsao-climatica-de-precipitacao-sobre-a-america-do-sul
now=$(date -u +%s); tgt=$(date -u -d 'tomorrow 00:00' +%s)
sleep $((tgt - now + 20))
sub() { $K competitions submit -c $C -f "submissions/$1" -m "$2" 2>&1; sleep 90; }
sub blend_v2_d80.csv        "blend 0.6*ridge_all+0.4*v2 + damping g=0.80, 108col - FINAL CAND 1"
sub blend_v2_65_d80.csv     "blend 0.65/0.35 + damping g=0.80 - peso sob damping"
sub blend_ridge_v2_65.csv   "blend 0.65/0.35 sem damping - isola efeito damping"
sub v05_ridge_all_d80.csv   "ridge_all puro + damping g=0.80 - FINAL CAND 2 (hedge)"
sub blend_v2_d70.csv        "blend 0.6/0.4 + damping g=0.70 - curva publica de gama"
