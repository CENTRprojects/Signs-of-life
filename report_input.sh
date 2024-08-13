#!/bin/bash
INPUT_FOLDER=$(perl -lne 'print $1 if /^input_folder=(.*)/' < .env)
rm $INPUT_FOLDER/*chunk*
echo
wc -l $INPUT_FOLDER*
echo
echo -e "folder\t$INPUT_FOLDER"
echo
echo -e "gtlds\t\t$(ls $INPUT_FOLDER | grep -E \\.[a-z][a-z][a-z_-]+\\. | grep -v covid | wc -l)  :  gtlds (only .name is outside CZDS)"
echo -e "idns\t\t$(ls $INPUT_FOLDER | grep -E \\.xn--[a-z0-9]+\\. | grep -v 'p1ai' | wc -l)   :  gtld idns only"
echo -e "covid\t\t$(ls $INPUT_FOLDER | grep -E \\.covid | wc -l)   :  covid specific domains"
echo -e "cctlds\t\t$(ls $INPUT_FOLDER | grep -E \\.[a-z][a-z]\\.\|xn--p1ai | wc -l)  :  including specific cctld idns"


echo --------------------
echo -e "total \t\t$(ls $INPUT_FOLDER | wc -l)"
echo
echo cctlds:
ls $INPUT_FOLDER | grep \\.[a-z][a-z]\\.
