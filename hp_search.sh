#!/bin/bash -e
mkdir -p tmp
for lr in 0.0001 0.00001; do
    for wd in 0.1 0.01 0.001; do
        for warmup_updates in 3000 10000; do
            for ga in 1 4; do
                work=$(mktemp -d --tmpdir=tmp/)
                mkdir -p $work
                echo "Submitted $work"
                qsub -N nnlm -S /bin/bash -l q_short_gpu -l 'h=vgn[ij]*' -cwd -V -P recapp -o $work/trainlog -e $work/trainlog run.sh \
                    lr=$lr wd=$wd warmup_updates=$warmup_updates grad_accum=$ga workd=$work
            done
        done
    done
done
