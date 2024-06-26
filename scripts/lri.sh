
# actstrack_2T
# GMT-lin
python trainer.py -ba --cuda 0 --backbone egnn --dataset actstrack_2T --method lri_bern -mt 3 -ie 0.1

# GMT-sam
python trainer.py -ba -smt 55 --cuda 0 --backbone egnn --dataset actstrack_2T --method lri_bern -st 1 -mt 55 -ir 1
python trainer.py -ba -smt 55 --cuda 0 --backbone egnn --dataset actstrack_2T --method lri_bern -st 1 -mt 5553 -sr 0.4

# tau3mu
# GMT-lin
python trainer.py -ba --cuda 0 --backbone egnn --dataset tau3mu --method lri_bern -mt 3 -ie 0.01

# GMT-sam
python trainer.py -ba -smt 88 --cuda 0 --backbone egnn --dataset tau3mu --method lri_bern -st 1 -mt 88 
python trainer.py -ba -smt 88 --cuda 0 --backbone egnn --dataset tau3mu --method lri_bern -st 1 -mt 5553 -sr 0.3

# synmol
# GMT-lin
python trainer.py -ba --cuda 0 --backbone egnn --dataset synmol --method lri_bern -mt 3 -ie 0.1

# GMT-sam
python trainer.py -ba -smt 55 -fr 0.8 --cuda 0 --backbone egnn --dataset synmol --method lri_bern -st 20 -mt 55 -ir 1
python trainer.py -ba -smt 55 -fr 0.8 --cuda 0 --backbone egnn --dataset synmol --method lri_bern -st 20 -mt 5550 -sr 0.6

# plbind
# GMT-lin
python trainer.py -ba --cuda 0 --backbone egnn --dataset plbind --method lri_bern -mt 3 -ie 0.1

# GMT-sam
python trainer.py -ba -smt 55 -ie 0.1 -fr 0.6 --cuda 0 --backbone egnn --dataset plbind --method lri_bern -st 1 -mt 55 -ir 1
python trainer.py -ba -smt 55 -ie 0.1 -fr 0.6 --cuda 0 --backbone egnn --dataset plbind --method lri_bern -st 1 -mt 5550 -sr 0.8
