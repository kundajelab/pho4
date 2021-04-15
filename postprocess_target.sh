#!/bin/bash

target=$1

cp /srv/scratch/amr1/chip-seq-pipeline2/chip/${target}/call-qc_report/execution/qc.html /srv/www/kundaje/amr1/pho4/qc_reports/gcpbm/${target}.html

if [ -d /srv/scratch/amr1/chip-seq-pipeline2/chip/${target}/call-count_signal_track_pooled ]; then
    cp /srv/scratch/amr1/chip-seq-pipeline2/chip/${target}/call-count_signal_track_pooled/execution/*bigwig /users/amr1/pho4/data/gcpbm/${target}/.
else
    cp /srv/scratch/amr1/chip-seq-pipeline2/chip/${target}/call-count_signal_track/shard-0/execution/*negative.bigwig /users/amr1/pho4/data/gcpbm/${target}/basename_prefix.pooled.negative.bigwig
    cp /srv/scratch/amr1/chip-seq-pipeline2/chip/${target}/call-count_signal_track/shard-0/execution/*positive.bigwig /users/amr1/pho4/data/gcpbm/${target}/basename_prefix.pooled.positive.bigwig
fi

cp /srv/scratch/amr1/chip-seq-pipeline2/chip/${target}/call-reproducibility_idr/execution/idr.optimal_peak.narrowPeak.gz /users/amr1/pho4/data/gcpbm/${target}/idr.optimal_peak.narrowPeak.gz

zcat /users/amr1/pho4/data/gcpbm/${target}/idr.optimal_peak.narrowPeak.gz | perl -lane 'print $F[0]."\t".(($F[1]+$F[9]))."\t".(($F[1]+$F[9]))' | bedtools slop -g /users/amr1/pho4/data/genome/hg38/hg38.chrom.sizes -b 500 | perl -lane 'if  ($F[2]-$F[1] == 1000) {print $F[0]."\t".$F[1]."\t".$F[2]."\t1"}' | sortBed | gzip -c > /users/amr1/pho4/data/gcpbm/${target}/1k_around_summits.bed.gz

zcat /users/amr1/pho4/data/gcpbm/${target}/1k_around_summits.bed.gz | egrep -w 'chr1|chr8|chr21' | gzip -c > /users/amr1/pho4/data/gcpbm/${target}/test_1k_around_summits.bed.gz

zcat /users/amr1/pho4/data/gcpbm/${target}/1k_around_summits.bed.gz | egrep -w 'chr22' | gzip -c > /users/amr1/pho4/data/gcpbm/${target}/valid_1k_around_summits.bed.gz

zcat /users/amr1/pho4/data/gcpbm/${target}/1k_around_summits.bed.gz | egrep -w -v 'chr1|chr8|chr21|chr22' | gzip -c > /users/amr1/pho4/data/gcpbm/${target}/train_1k_around_summits.bed.gz

if [ -d /srv/scratch/amr1/chip-seq-pipeline2/chip/${target}/call-align_ctl ]; then
    cp /srv/scratch/amr1/chip-seq-pipeline2/chip/${target}/call-align_ctl/*/execution/*bam /users/amr1/pho4/data/gcpbm/${target}/ctl/.
    samtools merge /users/amr1/pho4/data/gcpbm/${target}/ctl/control.bam /users/amr1/pho4/data/gcpbm/${target}/ctl/*.bam
    bedtools genomecov -5 -bg -strand + -g /users/amr1/pho4/data/genome/hg38/hg38.chrom.sizes -ibam /users/amr1/pho4/data/gcpbm/${target}/ctl/control.bam | sort -k1,1 -k2,2n > /users/amr1/pho4/data/gcpbm/${target}/ctl/control_pos_strand.bedGraph
    bedtools genomecov -5 -bg -strand - -g /users/amr1/pho4/data/genome/hg38/hg38.chrom.sizes -ibam /users/amr1/pho4/data/gcpbm/${target}/ctl/control.bam | sort -k1,1 -k2,2n > /users/amr1/pho4/data/gcpbm/${target}/ctl/control_neg_strand.bedGraph
    /users/amr1/pho4/data/genome/bedGraphToBigWig /users/amr1/pho4/data/gcpbm/${target}/ctl/control_pos_strand.bedGraph /users/amr1/pho4/data/genome/hg38/hg38.chrom.sizes /users/amr1/pho4/data/gcpbm/${target}/ctl/control_pos_strand.bw
    /users/amr1/pho4/data/genome/bedGraphToBigWig /users/amr1/pho4/data/gcpbm/${target}/ctl/control_neg_strand.bedGraph /users/amr1/pho4/data/genome/hg38/hg38.chrom.sizes /users/amr1/pho4/data/gcpbm/${target}/ctl/control_neg_strand.bw
fi