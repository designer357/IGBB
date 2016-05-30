#!/bin/bash
clear
echo "Good morning, world."
read -p "Press any key to START..."
#array=("[1,2]" "[2,2]" "3" "4" "5")
array_i=("2" "4" "6" "8" "10" "12" "14" "16" "18" "20" "22" "24" "26" "28" "30" "32" "34" "36" "38" "40" "42" "44" "46" "48" "50" "52" "54" "56" "58" "60" "62" "64" "66" "68" "70" "72" "74" "76" "78" "80")
#array_i=("60" "62" "64" "66" "68" "70" "72" "74" "76" "78" "80")
#array_i=("5" "10" "20" "25" "30")
#array_j=("5" "10" "20" "25" "30")

#array_j=("60" "62" "64" "66" "68" "70" "72" "74" "76" "78" "80")
#array_i=("2" "4" "6" "8" "10" "12" "14" "16" "18" "20" "22" "24" "26" "28" "30")

#array_j=("2" "4" "6" "8" "10" "12" "14" "16" "18" "20" "22" "24" "26" "28" "30")
array_j=("2" "4" "6" "8" "10" "12" "14" "16" "18" "20" "22" "24" "26" "28" "30" "32" "34" "36" "38" "40" "42" "44" "46" "48" "50" "52" "54" "56" "58" "60" "62" "64" "66" "68" "70" "72" "74" "76" "78" "80")
for i in  ${array_i[@]}
do
    for j in  ${array_j[@]}
        do
	    #printf "python ./Keras3.py "
    	#printf "i is $i, j is $j"
    	#printf "\n"
    	python ./Main_Keras2.py  $i $j &
    	wait
        done
done
#wait
read -p "Press any key to END..."
#echo "Do you wish to install this program?"
#select yn in "Yes" "No"; do
    #case $yn in
        #Yes ) make install; break;;
        #No ) exit;;
    #esac
#done





