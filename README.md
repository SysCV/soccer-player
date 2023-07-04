# VIS Project Template

## How to use the repo

- [vscode](https://code.visualstudio.com/) is recommended as an IDE. [.vscode](.vscode) folder provides a default setup that is consistent with our code style requirements.
- Install the proper python version on your system.
- Add your python dependencies to [`requirements.txt`](./requirements.txt). The default requirements install the tools for [static code check](<https://en.wikipedia.org/wiki/Lint_(software)>) and [unit tests](https://en.wikipedia.org/wiki/Unit_testing).
- Install dependencies

  ```bash
  pip install -r requirements
  ```

  It is important to install the dependences first so that your IDE can recognize packages and provide intelligent assistance.

- All the python code should go to [`proj_vis`](proj_vis/). It is a root package. Each sub package needs to have `__init__.py` in the sub folder, as required by python.
- Search for `proj_vis` globally and replace it with your project name. Also change the root package name to your project name.
- Images, assets, or other binary files should not be committed to git unless they are used for code configurations, examples and unit tests. They should be stored in the shared Google Drive folder.
- Our python code should follow the [standard python style guide](https://www.python.org/dev/peps/pep-0008/). Comments and doc string follow the [Google guide](https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings). You can use `black` and `isort` to automatically format your code, as in [`scripts/lint.sh`](./scripts/lint.sh). We also require `mypy` checking for our major open source projects. It checks python [type annotations](https://docs.python.org/3/library/typing.html) and make sure the code statements are properly connected. It is optional for student projects. An example of the python code is at [`proj_vis/demo.py`](proj_vis/demo.py). Here is an example in released package: [scalabel/label/io.py](https://github.com/scalabel/scalabel/blob/master/scalabel/label/io.py). We can ignore the type errors of the external packages. Those static code analysis can not only ensure consistent coding styles, but also catch a lot of errors and bugs without running the code.
- On your local machine, you can run

  ```bash
  sh scripts/lint.sh
  ```

  to perform static code analysis. It will run the following commands in order.

  ```bash
  python3 -m black proj_vis
  python3 -m isort proj_vis
  python3 -m pylint proj_vis
  python3 -m pydocstyle --convention=google proj_vis
  python3 -m mypy --install-types --strict proj_vis
  ```

- The correct way to run a script is to run it as a package. For example,

  ```bash
  python3 -m proj_vis.demo
  ```

  **Not**

  ```bash
  python3 proj_vis/demo.py
  ```

  The latter will make the python package hard to maintain.

- Provide unit tests for your core code. Our major open source code requires a full unit test coverage.
- We use `pytest` to run the unit tests. It is very easy to get started with [its doc](https://docs.pytest.org/). The test code for a file should locate in the same folder, with suffix "\_test.py". An example is [`proj_vis/demo_test.py`](proj_vis/demo_test.py). To run the tests, you can simply use

  ```bash
  python3 -m pytest --pyargs proj_vis
  ```

  This command will find all the test files in the package and run them.

- When you are ready or supposed to share your changes on GitHub, commit your changes in a branch other than `main` and push it to GitHub.
- Create a pull request (PR) with your branch. The PR can be a draft to show that you are still working on it. Draft PRs are where you can share your changes and progress, not branches.
- GitHub actions will always run continuous integration (CI) to check code style and run unit tests for each change of the PR.
- When you are ready to merge your code, make sure your code passes all the GitHub checks. Then ask someone to review your code. After someone approves the code, you can squash and merge your code to the main branch. It will live on GitHub happily ever after.

Copyright © 2021, [ETH VIS Group](https://cv.ethz.ch/).
