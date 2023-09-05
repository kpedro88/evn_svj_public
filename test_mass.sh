#!/bin/bash -x

DIR=sep5
SEED=10
for SAMPLE in dijetPair dijetPairLowMass dijetPairHighMass; do
	echo $SAMPLE
	if [ "$SAMPLE" = dijetPairHighMass ]; then SEED=10000; fi
	python train.py -C configs/${SAMPLE}.py --outf ${SAMPLE}_${DIR}_1 --epochs 500 --sampling undersample --learning-rate 0.005 --random-seed $SEED --best
	python train.py -C configs/${SAMPLE}.py --outf ${SAMPLE}_${DIR}_2 --epochs 500 --sampling undersample --learning-rate 0.001 --random-seed $SEED --best --continue-train ${SAMPLE}_${DIR}_1/models
	python test.py -C configs/${SAMPLE}.py --outf ${SAMPLE}_${DIR}_2 --name test --calibrate
	python test.py -C configs/${SAMPLE}.py --outf ${SAMPLE}_${DIR}_2 --name testGauss --calibrate --calib-gauss
	python analyze.py -C configs/dijetPairAll.py --outf ${SAMPLE}_${DIR}_2 --name plotsPair --theory "" --calib-source ${SAMPLE}_${DIR}_2/testGauss/calibrations.py --extras Mjj_msortedP1_high Mjj_msortedP1_low Mjj_msortedP2_high Mjj_msortedP2_low Mjj_msortedP3_high Mjj_msortedP3_low --pair
	python analyze.py -C configs/dijetPair.py --outf ${SAMPLE}_${DIR}_2 --name plotsPairAvg --theory dR_M Truth_M_avg Truth_high_M_avg Truth_avg_M_avg --xmin 250 --xmax 1250 --bins 50 --calib-source ${SAMPLE}_${DIR}_2/plotsPair/calibrations.py
	python analyze.py -C configs/dijetPair.py --outf ${SAMPLE}_${DIR}_2 --name plotsGaussAvg --theory dR_M Truth_M_avg Truth_high_M_avg Truth_avg_M_avg --xmin 250 --xmax 1250 --bins 50 --calib-source ${SAMPLE}_${DIR}_2/testGauss/calibrations.py
	python analyze.py -C configs/dijetPair.py --outf ${SAMPLE}_${DIR}_2 --name plotsPairMass --theory "" --extras Mjj_msortedP1_high Mjj_msortedP1_low Mjj_msortedP2_high Mjj_msortedP2_low Mjj_msortedP3_high Mjj_msortedP3_low --pair-mass --xmin 250 --xmax 1250 --bins 50 --calib-source ${SAMPLE}_${DIR}_2/plotsPair/calibrations.py -v
done
