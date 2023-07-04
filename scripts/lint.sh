python3 -m black proj_vis
python3 -m isort proj_vis
python3 -m pylint proj_vis
python3 -m pydocstyle --convention=google proj_vis
python3 -m mypy --install-types --strict proj_vis
