# Minimal makefile for Sphinx documentation
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Sphinx builders that need generated reference pages first.
SPHINX_TARGETS = html dirhtml singlehtml pickle json htmlhelp qthelp devhelp epub latex latexpdf text man changes linkcheck doctest coverage gettext

.PHONY: help Makefile clean reference/metrics.md $(SPHINX_TARGETS)

clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

reference/metrics.md: _generate_reference.py ../src/neurosetta/utils/metrics/registry.py
	python _generate_reference.py

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

$(SPHINX_TARGETS): reference/metrics.md
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
