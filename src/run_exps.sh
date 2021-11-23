for file in ../exps2run/*
do
    python pricing.py -c $file
done