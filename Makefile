dvc_push:
	dvc add mlruns
	dvc push
	git add .
	git commit -m "pushed to dvc"
	git push