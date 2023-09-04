#!/bin/bash -x

DIR=sep4
SEED=10
for SAMPLE in dijetPair dijetPairLowMass dijetPairHighMass; do
	echo $SAMPLE
	if [ "$SAMPLE" = dijetPairLowMass ]; then SEED=100; fi
	if [ "$SAMPLE" = dijetPairHighMass ]; then SEED=10000; fi
	python train.py -C configs/${SAMPLE}.py --outf ${SAMPLE}_${DIR}_1 --epochs 50 --sampling undersample --learning-rate 0.01 --random-seed $SEED
	python train.py -C configs/${SAMPLE}.py --outf ${SAMPLE}_${DIR}_2 --epochs 100 --sampling undersample --learning-rate 0.001 --random-seed $SEED --continue-train ${SAMPLE}_${DIR}_1/models
	python train.py -C configs/${SAMPLE}.py --outf ${SAMPLE}_${DIR}_3 --epochs 50 --sampling undersample --learning-rate 0.0001 --random-seed $SEED --continue-train ${SAMPLE}_${DIR}_2/models
	python test.py -C configs/${SAMPLE}.py --outf ${SAMPLE}_${DIR}_3 --name testGauss --calibrate --calib-gauss
	python analyze.py -C configs/dijetPairAll.py --outf ${SAMPLE}_${DIR}_3 --name plotsPair --theory "" --calib-source ${SAMPLE}_${DIR}_3/testGauss/calibrations.py --extras Mjj_msortedP1_high Mjj_msortedP1_low Mjj_msortedP2_high Mjj_msortedP2_low Mjj_msortedP3_high Mjj_msortedP3_low --pair
	python analyze.py -C configs/dijetPair.py --outf ${SAMPLE}_${DIR}_3 --name plotsPairAvg --theory dR_M Truth_M_avg Truth_high_M_avg Truth_avg_M_avg --xmin 250 --xmax 1250 --bins 50 --calib-source ${SAMPLE}_${DIR}_3/plotsPair/calibrations.py
done

