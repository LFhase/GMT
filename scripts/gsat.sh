
# BA-2Motifs

# GMT-lin
python run_gmt.py --dataset ba_2motifs --backbone GIN --cuda 0 -fs 1 -mt 3 -ie 0.5
# GMT-sam 
python run_gmt.py --dataset ba_2motifs --backbone GIN --cuda 0 -fs 1 -mt 8 -st 10 -ie 0.1 -r 0.5 -sm 
python run_gmt.py --dataset ba_2motifs --backbone GIN --cuda 0 -fs 1 -gmt 8 -st 10 -ie 0.1 -r 0.5 -fm -mt 5669


# GMT-lin
python run_gmt.py --dataset ba_2motifs --backbone PNA --cuda 0 -fs 1 -mt 3 -ie 0.1
# GMT-sam
python run_gmt.py --dataset ba_2motifs --backbone PNA --cuda 0 -fs 1 -mt 5 -st 20 -ie 0.1 -r 0.5 -sm 
python run_gmt.py --dataset ba_2motifs --backbone PNA --cuda 0 -fs 1 -gmt 5 -st 20 -ie 0.1 -r 0.5 -fm -mt 5669




# Mutag

# GMT-lin
python run_gmt.py --dataset mutag --backbone GIN --cuda 0 -fs 1 -mt 3 -ie 0.1
# GMT-sam
python run_gmt.py --dataset mutag --backbone GIN --cuda 0 -fs 1 -mt 5 -st 100 -ie 0.1 -sm 
python run_gmt.py --dataset mutag --backbone GIN --cuda 0 -fs 1 -gmt 5 -st 100 -ie 0.1 -fm -mt 5550


# GMT-lin 
python run_gmt.py --dataset mutag --backbone PNA --cuda 0 -fs 1 -mt 3 -ie 0.5
# GMT-sam
python run_gmt.py --dataset mutag --backbone PNA --cuda 0 -fs 1 -mt 5 -st 1 -ie 0.5 -r 0.6 -sm 
python run_gmt.py --dataset mutag --backbone PNA --cuda 0 -fs 1 -gmt 5 -st 1 -ie 0.5 -r 0.6 -fm -mt 5559





# MNIST-75sp

# GMT-lin
python run_gmt.py --dataset mnist --backbone GIN --cuda 0 -fs 1 -mt 3 -ie 1
# GMT-sam
python run_gmt.py --dataset mnist --backbone GIN --cuda 0 -fs 1 -mt 8 -st 100 -ie 0.1 -sm 
python run_gmt.py --dataset mnist --backbone GIN --cuda 0 -fs 1 -gmt 8 -st 100 -ie 0.1 -fm -mt 5550


# GMT-lin 
python run_gmt.py --dataset mnist --backbone PNA --cuda 0 -fs 1 -mt 3 -ie 1
# GMT-sam
python run_gmt.py --dataset mnist --backbone PNA --cuda 0 -fs 1 -mt 5 -st 1 -ie 0.1 -r 0.8 -sm 
python run_gmt.py --dataset mnist --backbone PNA --cuda 0 -fs 1 -gmt 5 -st 1 -ie 0.1 -r 0.8 -fm -mt 5552



# SPMotif-0.5

# GMT-lin
python run_gmt.py --dataset spmotif_0.7 --backbone GIN --cuda 0 -fs 1 -mt 3 -ie 0.5
# GMT-sam
python run_gmt.py --dataset spmotif_0.7 --backbone GIN --cuda 0 -fs 1 -mt 5 -di 10 -st 80 -ie 0.5 -sm 
python run_gmt.py --dataset spmotif_0.7 --backbone GIN --cuda 0 -fs 1 -gmt 5 -di 10 -st 80 -ie 0.5 -mt 5550 -sr 0.8


# GMT-lin 
python run_gmt.py --dataset spmotif_0.5 --backbone PNA --cuda 0 -fs 1 -mt 3 -ie 1
# GMT-sam
python run_gmt.py --dataset spmotif_0.5 --backbone PNA --cuda 0 -fs 1 -mt 5 -di -1 -st 80 -ie 1 -sm 
python run_gmt.py --dataset spmotif_0.5 --backbone PNA --cuda 0 -fs 1 -gmt 5 -di -1 -st 80 -ie 1 -mt 5552 -sr 0.3



# SPMotif-0.7

# GMT-lin
python run_gmt.py --dataset spmotif_0.7 --backbone GIN --cuda 0 -fs 1 -mt 3 -ie 0.5
# GMT-sam
python run_gmt.py --dataset spmotif_0.7 --backbone GIN --cuda 0 -fs 1 -mt 5 -di 20 -st 200 -ie 0.1 -sm 
python run_gmt.py --dataset spmotif_0.7 --backbone GIN --cuda 0 -fs 1 -gmt 5 -di 20 -st 200 -ie 0.1 -fm -mt 5550 -sr 0.7


# GMT-lin 
python run_gmt.py --dataset spmotif_0.7 --backbone PNA --cuda 0 -fs 1 -mt 3 -ie 1
# GMT-sam
python run_gmt.py --dataset spmotif_0.7 --backbone PNA --cuda 0 -fs 1 -mt 5 -di -1 -st 100 -ie 0.1 -r 0.8 -sm 
python run_gmt.py --dataset spmotif_0.7 --backbone PNA --cuda 0 -fs 1 -gmt 5 -di -1 -st 100 -ie 0.1 -r 0.8 -fm -mt 5552 -sr 0.3



# SPMotif-0.9

# GMT-lin
python run_gmt.py --dataset spmotif_0.9 --backbone GIN --cuda 0 -fs 1 -mt 3 -ie 0.5
# GMT-sam
python run_gmt.py --dataset spmotif_0.9 --backbone GIN --cuda 0 -fs 1 -mt 5 -di 20 -st 200 -ie 0.1 -sm 
python run_gmt.py --dataset spmotif_0.9 --backbone GIN --cuda 0 -fs 1 -gmt 5 -di 20 -st 200 -ie 0.1 -fm -mt 5550 -sr 0.5


# GMT-lin 
python run_gmt.py --dataset spmotif_0.9 --backbone PNA --cuda 0 -fs 1 -mt 3 -ie 1
# GMT-sam
python run_gmt.py --dataset spmotif_0.9 --backbone PNA --cuda 0 -fs 1 -mt 5 -di -1 -st 80 -ie 1 -sm 
python run_gmt.py --dataset spmotif_0.9 --backbone PNA --cuda 0 -fs 1 -gmt 5 -di -1 -st 80 -ie 1 -mt 5552 -sr 0.3



# Molhiv

# GMT-lin
python run_gmt.py --dataset ogbg_molhiv --backbone GIN --cuda 0 -fs 1 -mt 3 -ie 1
# GMT-sam
python run_gmt.py --dataset ogbg_molhiv --backbone GIN --cuda 0 -fs 1 -mt 5 -di 20 -st 100 -ie 0.1 -sm 
python run_gmt.py --dataset ogbg_molhiv --backbone GIN --cuda 0 -fs 1 -gmt 5 -di 20 -st 100 -ie 0.1 -fm -mt 5552 


# GMT-lin 
python run_gmt.py --dataset ogbg_molhiv --backbone PNA --cuda 0 -fs 1 -mt 3 -ie 1
# GMT-sam
python run_gmt.py --dataset ogbg_molhiv --backbone PNA --cuda 0 -fs 1 -mt 5 -di 10 -st 100 -ie 0.5 -sm 
python run_gmt.py --dataset ogbg_molhiv --backbone PNA --cuda 0 -fs 1 -gmt 5 -di 10 -st 100 -ie 0.5 -fm -mt 5449


# Graph-SST2

# GMT-lin
python run_gmt.py --dataset ogbg_molhiv --backbone GIN --cuda 0 -fs 1 -mt 3 -ie 1
# GMT-sam 
python run_gmt.py --dataset ogbg_molhiv --backbone GIN --cuda 0 -fs 1 -mt 5 -r 0.7 -st 100 -ie 1 -sm 
python run_gmt.py --dataset ogbg_molhiv --backbone GIN --cuda 0 -fs 1 -gmt 5 -r 0.7 -st 100 -ie 1 -fm -mt 5669 


# GMT-lin 
python run_gmt.py --dataset ogbg_molhiv --backbone PNA --cuda 0 -fs 1 -mt 3 -ie 0.1
# GMT-sam 
python run_gmt.py --dataset ogbg_molhiv --backbone PNA --cuda 0 -fs 1 -mt 5 -di 20 -st 100 -ie 0.5 -sm 
python run_gmt.py --dataset ogbg_molhiv --backbone PNA --cuda 0 -fs 1 -gmt 5 -di 20 -st 100 -ie 0.5 -fm -mt 5669
