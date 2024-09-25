
.PHONY: assemble package release clean upload

#Default to what's on the path unless set in an env var
PIP3 ?= pip3
PYTHON3 ?= python3

tags:
	etags *.py */*py



package: clean prereqs
	$(PYTHON3) -m build -n

prereqs:
#Note: We install pkginfo 1.8.3 because 1.9.0 is broken.  When 1.9.1 is available it can be used

	$(PIP3) install --upgrade pip build wheel setuptools pkginfo==1.9.6

install: package
	$(PIP3) install --force-reinstall dist/*.whl

venv: 
	rm -rf virtualenv
	rm -f virtualenv.tgz
	$(PYTHON3) -m venv virtualenv
	. virtualenv/bin/activate && pip3 install dist/*.whl && pip3 install venv-pack && venv-pack -o virtualenv.tgz

	

upload: package
	$(PIP3) install --upgrade twine
	$(PYTHON3) -m twine upload --verbose --skip-existing --repository-url ${NEXUS_REPO} dist/*

clean:
	rm -rf dist

