setup:
	python3 -m venv venv-octopus
	venv-octopus/bin/pip install -r requirements.txt

convert-images:
	venv-octopus/bin/python app.py

clean:
	rm -rf *.pyc