files=`ls *.csv | paste -s -d ',' -`
python3 plot.py $files
