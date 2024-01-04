run:
	exp_run
	dvc_push

exp_run:
	./bash_scripts/run_grembedding.sh

dvc_push:
	./bash_scripts/after-dvc.sh