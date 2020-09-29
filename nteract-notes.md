Get nteract running:

	https://nteract.io/kernels
	python3 -m venv my_environment_name      # create a virtual environment
	source my_environment_name/bin/activate  # activate the virtual environment
	python -m pip install ipykernel          # install the python kernel (ipykernel) into the virtual environment
	python -m ipykernel install              # install python kernel into nteract's available kernel list