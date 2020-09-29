eeprivacy
=========

This repository contains the pilot implementation of the core privacy methods for the Energy Data Vault. The key components are:

* Core Differential Privacy for energy efficiency analytics (`eeprivacy`)
* Python API documentation for `eeprivacy`
* Sample implementations of key use cases

---

The Energy Data Vault (EDV) enables the use of the gold standard of privacy protection, differential privacy, for high value energy efficiency analytics. 

---

Development
-----------

Build docs:

	./bin/build_docs

Run tests:
	
	./bin/test

Get nteract running:

	https://nteract.io/kernels
	python3 -m venv my_environment_name      # create a virtual environment
	source my_environment_name/bin/activate  # activate the virtual environment
	python -m pip install ipykernel          # install the python kernel (ipykernel) into the virtual environment
	python -m ipykernel install              # install python kernel into nteract's available kernel list