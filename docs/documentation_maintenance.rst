======================================
Maintaining this Documentation
======================================

This documentation is built with `Sphinx <http://sphinx-doc.org/>`_, and 
hosted on github as a `github page <http://pages.github.com/>`_ for the lazyflow repo.

To make a changes to the docs, edit the desired .rst file(s) in the docs directory, and then build the docs:

.. code-block:: bash

 	$ cd docs
 	$ make html

First, view your changes locally:

.. code-block:: bash

	$ firefox _build/html/index.html

Your changes will not be visible online until they are applied to the special ``gh-pages`` branch of lazyflow and pushed.

There is a script in lazyflow for automating this process.
It is highly recommended that you use a special local copy of the lazyflow repo to do this.  Just follow these steps:

1) Make sure your changes to the .rst files are pushed to lazyflow/master.
2) Make a new clone of the lazyflow repo, and checkout the ``gh-pages`` branch.
3) Run the ``update_from_master.sh`` script.

Here's a walk-through (output not shown).

.. code-block:: bash

	$ pwd
	/home/bergs/workspace/lazyflow/docs
	$ git add -u .
	$ git commit -m "docs: Added instructions for documentation maintenance."
	$ git push origin master
	$ cd /tmp
	$ git clone ssh://git@github.com/ilastik/lazyflow lazyflow-gh-pages
	$ cd lazyflow-gh-pages/
	$ git checkout gh-pages
	$ ./update_from_master.sh 

The ``update_from_master.sh`` script handles the necessary pre-processing required by the github pages system.
You view the updated documentation at `<http://ilastik.github.com/lazyflow>`_.
