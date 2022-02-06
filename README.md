# Challenge_plancton
This is a challenge on kaggle. 

Lien pour tuto connexion DCEJS (GPU)

https://teaching.pages.centralesupelec.fr/deeplearning-lectures-build/cluster.html


Path pour les images sur les serveurs de CS : 
/mounts/Datasets1/ChallengeDeep/train/

#!/bin/bash

#File: tree-md

tree=$(tree -tf --noreport -I '*~' --charset ascii  |
       sed -e 's/| \+/  /g' -e 's/[|`]-\+/ */g' -e 's:\(* \)\(\(.*/\)\([^/]\+\)\):[]():g')

printf "# Project tree\n\n${tree}"
